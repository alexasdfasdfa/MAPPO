import asyncio
import os
import re
import threading
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import queue
import traceback
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


# ------------------ 模型调用与点集提取 ------------------
async def ask_model_free(
    prompt: str,
    get_credentials_func=None,   # 新增：用于获取登录凭证的函数
    headless: bool = True,       # 默认无头模式，不弹出浏览器窗口
    user_data_dir: str = None,
    keep_browser_open: bool = False,
) -> str:
    if user_data_dir is None:
        user_data_dir = os.path.join(os.getcwd(), "edge_profile")

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir,
            channel="msedge",
            headless=headless,
            args=[]
        )
        page = await context.new_page()

        try:
            # 1. 访问网站
            await page.goto("https://www.rsk.cn/", wait_until="domcontentloaded")

            # 2. 登录检测（用户头像）
            try:
                await page.wait_for_selector('button.rounded-full.bg-\\[var\\(--primary\\)\\]', timeout=5000)
                print("检测到用户头像，已登录状态。")
            except PlaywrightTimeoutError:
                print("未检测到用户头像，尝试自动登录...")
                if get_credentials_func is None:
                    raise Exception("未登录且未提供登录凭证获取函数，无法继续。")
                
                # 调用 GUI 获取用户名密码
                username, password = get_credentials_func()
                if not username or not password:
                    raise Exception("用户名或密码为空，登录失败。")

                # ----- 自动登录流程（需根据实际网站调整）-----
                # 示例：假设网站有“登录”按钮，点击后弹出表单
                try:
                    # 尝试点击“登录”按钮（选择器需实际测试）
                    await page.click('button:has-text("登录")', timeout=5000)
                except PlaywrightTimeoutError:
                    # 可能直接有输入框，跳过
                    pass
                
                # 等待用户名输入框出现（选择器需根据网站调整）
                try:
                    await page.wait_for_selector('input[name="username"], input[type="text"]', timeout=10000)
                except PlaywrightTimeoutError:
                    raise Exception("未找到用户名输入框，请检查选择器。")
                
                # 填写用户名
                await page.fill('input[name="username"]', username)
                # 填写密码
                await page.fill('input[name="password"]', password)
                # 点击登录按钮
                await page.click('button[type="submit"], button:has-text("登录")')
                
                # 等待登录成功（头像出现）
                await page.wait_for_selector('button.rounded-full.bg-\\[var\\(--primary\\)\\]', timeout=15000)
                print("自动登录成功。")

            # 3. 等待输入框并填入问题
            await page.fill("textarea", prompt)

            # 4. 记录当前消息数量
            initial_count = await page.evaluate('''
                () => document.querySelectorAll('div.flex-1.min-w-0.overflow-x-hidden.rounded-2xl').length
            ''')
            # 记录最后一条消息的初始文本
            initial_last_text = await page.evaluate('''
                () => {
                    const msgs = document.querySelectorAll('div.flex-1.min-w-0.overflow-x-hidden.rounded-2xl');
                    return msgs.length ? msgs[msgs.length-1].innerText : '';
                }
            ''')

            # 5. 发送问题
            await page.keyboard.press("Enter")

            # 6. 等待新消息出现（至少增加1条，即用户消息）
            await page.wait_for_function(
                """
                (initialCount) => {
                    const messages = document.querySelectorAll('div.flex-1.min-w-0.overflow-x-hidden.rounded-2xl');
                    return messages.length > initialCount;
                }
                """,
                arg=initial_count,
                timeout=10000
            )
            print("检测到新消息（用户消息已出现）。")

            # 7. 等待停止按钮出现并消失（确保回复生成完毕）
            try:
                await page.wait_for_selector('button:has-text("停止")', timeout=5000)
                print("检测到停止按钮，AI正在生成...")
                await page.wait_for_selector('button:has-text("停止")', state='hidden', timeout=60000)
                print("停止按钮消失，生成完成。")
            except PlaywrightTimeoutError:
                print("未检测到停止按钮，直接等待助手消息。")

            # 8. 等待最后一条消息的内容发生变化（助手消息出现）
            await page.wait_for_function(
                """
                (initialLastText) => {
                    const messages = document.querySelectorAll('div.flex-1.min-w-0.overflow-x-hidden.rounded-2xl');
                    if (messages.length === 0) return false;
                    const lastMsg = messages[messages.length-1];
                    return lastMsg.innerText.trim() !== initialLastText;
                }
                """,
                arg=initial_last_text,
                timeout=100000
            )
            print("助手消息已出现。")

            # 9. 提取最新回复内容
            answer_text = None
            for retry in range(3):
                messages = await page.query_selector_all('div.flex-1.min-w-0.overflow-x-hidden.rounded-2xl')
                if not messages:
                    await asyncio.sleep(1)
                    continue
                last_msg = messages[-1]
                text = await last_msg.inner_text()
                if not text.strip() or text.strip() == prompt.strip():
                    # 如果还是用户消息，再等一下
                    await asyncio.sleep(1)
                    continue
                answer_text = text.strip()
                break

            if answer_text is None:
                raise Exception("未能成功提取回复内容")

            # 10. 如果要求保持浏览器打开，则等待用户手动关闭
            if keep_browser_open and not headless:
                print("\n答案已获取。浏览器将保持打开，按 Enter 键关闭浏览器并继续...")
                input()

            return answer_text

        except Exception as e:
            print("调用出错，完整错误信息：")
            traceback.print_exc()
            try:
                if page and not page.is_closed():
                    await page.screenshot(path="error_screenshot.png")
                    print("已保存错误截图 error_screenshot.png")
            except Exception:
                pass
            return f"Error: {str(e)}"

        finally:
            await context.close()


def extract_points_from_text(text, debug=False):
    """从文本中提取点集坐标，返回列表[(x,y), ...]"""
    # 多种前缀模式（支持中英文）
    prefixes = [
        r'\*\*点集\s*[=：:]\s*',   # **点集 =**
        r'点集\s*[=：:]\s*',        # 点集 =
        r'```\s*点集\s*[=：:]\s*', # 代码块中
        r'\*\*Point set\s*[=：:]\s*',  # **Point set =**
        r'Point set\s*[=：:]\s*',      # Point set =
    ]
    for prefix in prefixes:
        pattern = prefix + r'\{((?:\([^)]+\)\s*,?\s*)+)\}(?:\*\*|```)?'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if debug:
                print(f"匹配到格式: {prefix}")
            points_str = match.group(1)
            coord_pattern = r'\((\d+),\s*(\d+)\)'
            points = [(int(x), int(y)) for x, y in re.findall(coord_pattern, points_str)]
            return points

    # 没有前缀，匹配最长的花括号块
    pattern_block = r'\{((?:\([^)]+\)\s*,?\s*)+)\}'
    matches = list(re.finditer(pattern_block, text, re.DOTALL))
    if matches:
        best_match = max(matches, key=lambda m: len(m.group(0)))
        if debug:
            print(f"未找到前缀，使用最长花括号块，长度 {len(best_match.group(0))}")
        points_str = best_match.group(1)
        coord_pattern = r'\((\d+),\s*(\d+)\)'
        points = [(int(x), int(y)) for x, y in re.findall(coord_pattern, points_str)]
        return points

    # 回退：匹配所有坐标
    if debug:
        print("未找到独立点集块，回退到全文本匹配所有坐标")
    all_points = re.findall(r'\((\d+),\s*(\d+)\)', text)
    return [(int(x), int(y)) for x, y in all_points]


# ------------------ GUI 应用 ------------------
class PointGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Set Generator")
        self.root.geometry("550x600")
        self.points = None
        self.current_figure = None
        self.canvas = None
        self.login_queue = queue.Queue()   # 用于跨线程传递登录凭证

        self.create_widgets()

    def create_widgets(self):
        # 顶部框架：横轴、纵轴
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10, padx=10, fill=tk.X)

        # 横轴距离
        tk.Label(top_frame, text="X range (min, max):").grid(row=0, column=0, padx=5)
        self.x_min_var = tk.StringVar(value="0")
        self.x_max_var = tk.StringVar(value="15")
        tk.Entry(top_frame, textvariable=self.x_min_var, width=8).grid(row=0, column=1, padx=2)
        tk.Entry(top_frame, textvariable=self.x_max_var, width=8).grid(row=0, column=2, padx=2)

        # 纵轴距离
        tk.Label(top_frame, text="Y range (min, max):").grid(row=0, column=3, padx=5)
        self.y_min_var = tk.StringVar(value="0")
        self.y_max_var = tk.StringVar(value="30")
        tk.Entry(top_frame, textvariable=self.y_min_var, width=8).grid(row=0, column=4, padx=2)
        tk.Entry(top_frame, textvariable=self.y_max_var, width=8).grid(row=0, column=5, padx=2)

        # 中间框架：点数和描述
        mid_frame = tk.Frame(self.root)
        mid_frame.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(mid_frame, text="Number of points:").pack(side=tk.LEFT, padx=5)
        self.points_count_var = tk.StringVar()
        tk.Entry(mid_frame, textvariable=self.points_count_var, width=10).pack(side=tk.LEFT, padx=5)

        tk.Label(mid_frame, text="Description:").pack(side=tk.LEFT, padx=5)
        self.desc_var = tk.StringVar()
        tk.Entry(mid_frame, textvariable=self.desc_var, width=30).pack(side=tk.LEFT, padx=5)

        # 底部按钮：生成、保存
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(pady=10)

        self.generate_btn = tk.Button(bottom_frame, text="Generate", command=self.on_generate, width=10)
        self.generate_btn.pack(side=tk.LEFT, padx=10)

        self.save_btn = tk.Button(bottom_frame, text="Save", command=self.on_save, width=10)
        self.save_btn.pack(side=tk.LEFT, padx=10)

        # 状态栏
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 绘图区域框架
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.init_plot()

    def init_plot(self):
        """Initialize an empty plot"""
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 30)
        ax.grid(True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        self.current_figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self, points, x_range, y_range):
        """Update plot with points"""
        if self.current_figure is not None:
            plt.close(self.current_figure)
        fig, ax = plt.subplots(figsize=(6, 5))
        if points:
            xs, ys = zip(*points)
            ax.scatter(xs, ys, c='red', s=50, zorder=5)
            for i, (x, y) in enumerate(points):
                ax.annotate(str(i+1), (x, y), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.grid(True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Point Set ({len(points)} points)")
        self.current_figure = fig
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _show_login_dialog(self):
        """在主线程中显示登录对话框，并将结果放入队列"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Login")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Username:").pack(pady=5)
        username_entry = tk.Entry(dialog, width=30)
        username_entry.pack()

        tk.Label(dialog, text="Password:").pack(pady=5)
        password_entry = tk.Entry(dialog, show="*", width=30)
        password_entry.pack()

        result = [None, None]

        def on_ok():
            result[0] = username_entry.get()
            result[1] = password_entry.get()
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="OK", command=on_ok, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=on_cancel, width=8).pack(side=tk.LEFT, padx=5)

        dialog.wait_window()
        self.login_queue.put(tuple(result))

    def get_credentials(self):
        """供子线程调用，返回 (username, password)"""
        self.root.after(0, self._show_login_dialog)
        try:
            username, password = self.login_queue.get(timeout=60)
        except queue.Empty:
            raise Exception("登录超时")
        return username, password

    def on_generate(self):
        """Generate button callback"""
        try:
            x_min = float(self.x_min_var.get())
            x_max = float(self.x_max_var.get())
            y_min = float(self.y_min_var.get())
            y_max = float(self.y_max_var.get())
            points_num = int(self.points_count_var.get())
            desc = self.desc_var.get().strip()
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure ranges are numbers and point count is integer.")
            return

        if x_min >= x_max or y_min >= y_max:
            messagebox.showerror("Range Error", "Minimum value must be less than maximum.")
            return

        if points_num <= 0:
            messagebox.showerror("Point Count Error", "Point count must be positive.")
            return

        prompt = f"""
In the plane with x ∈ [{x_min}, {x_max}] and y ∈ [{y_min}, {y_max}], use EXACTLY {points_num} integer points to form the outline of {desc}.
Please strictly output according to the following requirements:
1. Explain which parts the outline consists of (optional).
2. Finally, strictly output all point coordinates in the following format for program parsing. The point set MUST contain exactly {points_num} points.
  **Point set = {{(x1,y1), (x2,y2), ...}}**
Note: Coordinates must be integers, parentheses are English half-width, and coordinates are separated by commas and spaces.
"""
        print(prompt)
        self.generate_btn.config(state=tk.DISABLED)
        self.status_var.set("Generating...")
        self.root.update()

        def run_async_task():
            # 传入获取凭证的函数
            result = asyncio.run(ask_model_free(
                prompt,
                get_credentials_func=self.get_credentials,
                headless=True,      # 无头模式，不显示浏览器
                keep_browser_open=False
            ))
            self.root.after(0, self.on_model_result, result, (x_min, x_max, y_min, y_max))

        thread = threading.Thread(target=run_async_task, daemon=True)
        thread.start()

    def on_model_result(self, result_text, ranges):
        """Handle model response"""
        self.generate_btn.config(state=tk.NORMAL)
        if result_text.startswith("Error:"):
            self.status_var.set("Generation failed")
            messagebox.showerror("Model Error", f"Call failed: {result_text}")
            return

        points = extract_points_from_text(result_text, debug=True)
        if not points:
            self.status_var.set("No points extracted")
            messagebox.showerror("Parsing Error", "Failed to extract point set from model response.")
            print("Model response:\n", result_text)
            return

        self.points = points
        self.status_var.set(f"Successfully generated {len(points)} points")
        x_range = (ranges[0], ranges[1])
        y_range = (ranges[2], ranges[3])
        self.update_plot(points, x_range, y_range)

    def on_save(self):
        """保存当前点集到 dataset/{len(points)}.json，自动处理名称重复"""
        if self.points is None:
            messagebox.showinfo("Info", "No point set to save.")
            return

        n = len(self.points)
        desc = self.desc_var.get().strip()
        if not desc:
            desc = f"outline of letter 'A' (but actually {n} points)"  # 默认描述

        # 确保 dataset 目录存在
        dataset_dir = os.path.join(os.getcwd(), "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        json_file = os.path.join(dataset_dir, f"{n}.json")

        # 读取现有数据
        data = []
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
                except json.JSONDecodeError:
                    data = []

        # 收集已存在的名称
        existing_names = [rec.get("name", "") for rec in data]

        # 处理名称重复
        final_name = desc
        if desc in existing_names:
            counter = 2
            while f"{desc}_{counter}" in existing_names:
                counter += 1
            final_name = f"{desc}_{counter}"

        # 构造记录
        record = {
            "name": final_name,
            "len": n,
            "coordinates": [[x, y] for x, y in self.points]
        }

        # 追加新记录
        data.append(record)

        # 写回文件（紧凑格式，无缩进）
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

        messagebox.showinfo("Success", f"Point set saved to {json_file}\nName: {final_name}")
        self.status_var.set(f"Saved to {json_file} (name: {final_name})")


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PointGeneratorApp(root)
    root.mainloop()