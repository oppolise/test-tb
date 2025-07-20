# วิธีการ Debug โค้ด PyTorch Profiler Plugin ด้วย VS Code

เอกสารนี้อธิบายวิธีการ debug โค้ดในโฟลเดอร์ `tb_plugin/torch_tb_profiler/profiler/` โดยใช้ Visual Studio Code debugger. เนื่องจากการ debug plugin ที่ทำงานร่วมกับ web server อย่าง TensorBoard และต้องใช้ไฟล์ trace เป็น input มีความซับซ้อน, เอกสารนี้จะนำเสนอ 2 แนวทางหลัก:

1.  **การ Debug แบบ Standalone**: วิธีที่ง่ายและรวดเร็วที่สุดสำหรับทำความเข้าใจและทดสอบตรรกะของ parser แต่ละตัวโดยไม่ต้องรัน TensorBoard. **(แนะนำเป็นอย่างยิ่งสำหรับเริ่มต้น)**
2.  **การ Debug ร่วมกับ TensorBoard**: วิธีสำหรับทดสอบการทำงานร่วมกัน (integration) ในสภาพแวดล้อมจริงโดยการ "attach" debugger เข้ากับ process ของ TensorBoard.

---

## แนวทางที่ 1: การ Debug แบบ Standalone

แนวทางนี้คือการสร้าง script Python แยกต่างหากเพื่อจำลองการโหลดข้อมูลและเรียกใช้ parser ที่เราสนใจโดยตรง ทำให้เราสามารถใช้ VS Code debugger (กด F5) ได้ตามปกติ.

### ขั้นตอนการสร้าง Script Debug

1.  **สร้างไฟล์ใหม่**: สร้างไฟล์ Python ใหม่ใน root directory ของโปรเจกต์ (ระดับเดียวกับ `tb_plugin/`) และตั้งชื่อว่า `debug_standalone.py`.

2.  **เขียนโค้ดใน `debug_standalone.py`**:
    *   **Import ที่จำเป็น**:
        ```python
        import os
        # Import คลาสหลักที่ใช้ในการประมวลผลข้อมูล
        from tb_plugin.torch_tb_profiler.profiler.data import RunProfileData
        # Import คลาสสำหรับจัดการ cache (จำเป็นสำหรับ RunProfileData.parse)
        from tb_plugin.torch_tb_profiler.io.cache import Cache
        ```

    *   **ระบุไฟล์ Trace เป้าหมาย**: เลือกไฟล์ trace ที่ต้องการจะใช้ทดสอบ. ใน repository มีไฟล์ตัวอย่างอยู่ที่ `tb_plugin/samples/`.
        ```python
        # สร้าง path ที่ถูกต้องไปยังไฟล์ trace
        # __file__ คือ path ของ script ปัจจุบัน (debug_standalone.py)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # ตัวอย่างการใช้ไฟล์ trace ที่มีอยู่ (สามารถเปลี่ยนเป็นไฟล์อื่นได้)
        trace_path = os.path.join(script_dir, 'tb_plugin/samples/resnet50_num_workers_0/worker0.1623143089861.pt.trace.json.gz')
        ```

    *   **จำลองการทำงานและโหลดข้อมูล**: เขียนโค้ดเพื่อเรียก `RunProfileData.parse()` ซึ่งเป็นจุดเริ่มต้นของการประมวลผลทั้งหมด.
        ```python
        # RunProfileData.parse ต้องการ cache_dir เราสามารถสร้างโฟลเดอร์ temp ขึ้นมาได้
        cache_dir = os.path.join(script_dir, 'temp_cache')
        os.makedirs(cache_dir, exist_ok=True)

        print(f"Loading and parsing trace file: {os.path.basename(trace_path)}")
        # เรียก RunProfileData.parse โดยตรง
        # นี่คือขั้นตอนที่จำลองการทำงานของ `_process_data` ใน `loader.py`
        # เมื่อบรรทัดนี้ทำงานเสร็จ `data` object จะมีข้อมูลที่ผ่านการ process ทั้งหมดแล้ว
        # (เช่น operator tree, step breakdown, kernel stats, etc.)
        data = RunProfileData.parse(
            worker="worker0",          # ใส่ชื่อ worker สมมติ
            span=None,                 # ใส่ span สมมติ หรือ None
            path=trace_path,
            cache_dir=cache_dir
        )
        print("Parsing complete.")
        ```

    *   **เข้าถึงและทดสอบส่วนที่ต้องการ**: เมื่อได้ `data` object มาแล้ว เราสามารถเข้าถึงผลลัพธ์จาก parser ทุกตัวที่ถูกเรียกใน `data.process()` ได้เลย.
        ```python
        # --- ใส่โค้ดสำหรับทดสอบหรือตรวจสอบข้อมูลตรงนี้ ---

        # ตัวอย่างที่ 1: ตรวจสอบผลลัพธ์จาก EventParser และ OpTreeBuilder
        print(f"\n--- Testing EventParser/OpTreeBuilder results ---")
        if data.tid2tree:
            # tid2tree คือผลลัพธ์หลักของ OpTreeBuilder ที่ถูกเรียกโดย EventParser
            main_thread_id = list(data.tid2tree.keys())[0]
            root_node = data.tid2tree[main_thread_id]
            print(f"Operator tree generated for TID {main_thread_id}.")
            print(f"Root node has {len(root_node.children)} children.")
            # !!! ตั้ง breakpoint ที่นี่เพื่อสำรวจ `root_node` และโครงสร้าง tree ภายใน !!!
            pass # ใส่ pass เพื่อให้มีบรรทัดสำหรับตั้ง breakpoint

        # ตัวอย่างที่ 2: ตรวจสอบผลลัพธ์จาก StepParser
        if data.role_ranges:
            from tb_plugin.torch_tb_profiler.profiler.event_parser import ProfileRole
            kernel_time_ranges = data.role_ranges[ProfileRole.Kernel]
            comm_time_ranges = data.role_ranges[ProfileRole.Communication]
            print(f"Found {len(kernel_time_ranges)} kernel time ranges.")
            print(f"Found {len(comm_time_ranges)} communication time ranges.")
            # !!! ตั้ง breakpoint ที่นี่เพื่อดูค่าใน role_ranges !!!
            pass

        # ตัวอย่างที่ 3: ตรวจสอบผลลัพธ์จาก KernelParser
        print(f"\n--- Testing KernelParser results ---")
        if data.kernel_stat is not None:
            print(f"Kernel statistics generated for {len(data.kernel_stat)} unique kernels.")
            # kernel_stat เป็น pandas DataFrame
            print("Top 5 kernels by total duration:")
            print(data.kernel_stat.head(5))
            # !!! ตั้ง breakpoint ที่นี่เพื่อสำรวจ DataFrame `data.kernel_stat` !!!
            pass

        print("\nDebug script finished.")
        ```

### การใช้งานกับ VS Code Debugger

1.  **ตั้ง Breakpoint**: เปิดไฟล์ที่คุณต้องการจะ debug จริงๆ (เช่น `tb_plugin/torch_tb_profiler/profiler/event_parser.py` ในเมธอด `_parse_node`) แล้วคลิกที่ข้างซ้ายของหมายเลขบรรทัดเพื่อตั้ง breakpoint (จุดสีแดง).
2.  **เริ่ม Debug**: กลับไปที่ไฟล์ `debug_standalone.py` แล้วกด **F5**.
3.  Debugger จะเริ่มทำงาน, รัน script ของคุณ, และเมื่อการทำงานไปถึงบรรทัดที่คุณตั้ง breakpoint ไว้ (ไม่ว่าจะอยู่ในไฟล์ไหน), มันจะหยุดให้คุณตรวจสอบค่าตัวแปร, call stack, และไล่การทำงานทีละบรรทัดได้.

---

## แนวทางที่ 2: การ Debug ร่วมกับ TensorBoard (Attach to Process)

แนวทางนี้ซับซ้อนกว่า แต่มีประโยชน์สำหรับการทดสอบ integration และการ debug ปัญหาที่เกิดขึ้นเฉพาะเมื่อรันผ่าน TensorBoard เท่านั้น.

### ขั้นตอนการ Debug

1.  **รัน TensorBoard จาก Terminal**:
    *   Activate Python environment ที่ติดตั้ง `torch-tb-profiler` ไว้.
    *   ใช้คำสั่ง `tensorboard --logdir <path_to_your_log_dir>` เพื่อเริ่ม TensorBoard.
    *   **หาและจด Process ID (PID)** ของ TensorBoard.
        *   **Linux/macOS**: `ps aux | grep tensorboard`
        *   **Windows**: `tasklist | findstr "tensorboard"` หรือใช้ Task Manager.

2.  **ตั้งค่า `launch.json` ใน VS Code**:
    *   ไปที่ Debug view (ไอคอนรูปแมลง), คลิกที่ไอคอนรูปเฟือง และเลือก **Python**.
    *   เพิ่ม configuration ใหม่เข้าไปในไฟล์ `.vscode/launch.json`:
        ```json
        {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Python: Attach to TensorBoard",
                    "type": "python",
                    "request": "attach",
                    "processId": "${command:pickProcess}",
                    "justMyCode": false
                }
                // คุณสามารถเพิ่ม config สำหรับ debug subprocess ได้ที่นี่ (ดูหมายเหตุด้านล่าง)
            ]
        }
        ```
    *   **คำอธิบาย**:
        *   `"request": "attach"`: บอกให้ debugger ไป "เกาะ" กับ process ที่มีอยู่แล้ว.
        *   `"processId": "${command:pickProcess}"`: ทำให้ VS Code แสดง list ของ process ทั้งหมดให้เราเลือกเมื่อเริ่ม debug.
        *   `"justMyCode": false`: **สำคัญมาก!** ต้องตั้งเป็น `false` เพื่อให้ debugger สามารถหยุดในโค้ดของ library (`torch-tb-profiler`) ได้.

3.  **เริ่มการ Debug**:
    *   **ตั้ง Breakpoint**: เปิดไฟล์ใน `profiler/` ที่คุณต้องการ debug (เช่น `loader.py` ในเมธอด `load`) แล้วตั้ง breakpoint.
    *   **เลือก Configuration**: ใน Debug view, เลือก "Python: Attach to TensorBoard" จาก drop-down list.
    *   **กด F5 (Start Debugging)**.
    *   VS Code จะแสดง list ของ process, ให้ **ค้นหาและเลือก PID ของ TensorBoard** ที่คุณจดไว้.
    *   Debugger จะทำการ attach.

4.  **Trigger การทำงาน**:
    *   เปิด web browser แล้วไปที่ URL ของ TensorBoard (ปกติคือ `http://localhost:6006/`).
    *   ไปที่แท็บ "PyTorch Profiler". การคลิกที่แท็บนี้ หรือการเลือก run จาก drop-down list ใน UI จะเป็นการ trigger ให้ TensorBoard เรียกโค้ดใน plugin ของเรา.

5.  **Debugger หยุดที่ Breakpoint**:
    *   เมื่อโค้ดรันมาถึงบรรทัดที่คุณตั้ง breakpoint ไว้, VS Code จะหยุดการทำงานให้คุณ.

### การจัดการกับ Multiprocessing ใน `loader.py`

`RunLoader` ใช้ `multiprocessing`. การ attach แบบปกติจะอยู่แค่ใน main process. ถ้า breakpoint ของคุณอยู่ใน `_process_data` (ซึ่งรันใน child process), debugger อาจจะไม่หยุด.

*   **วิธีแก้ (Advanced)**: เพิ่ม `"subProcess": true` ใน `launch.json` configuration ของคุณ.
    ```json
    {
        "name": "Python: Attach (with Subprocess)",
        "type": "python",
        "request": "attach",
        "processId": "${command:pickProcess}",
        "justMyCode": false,
        "subProcess": true
    }
    ```
*   **หมายเหตุ**: การ debug subprocess อาจไม่เสถียรเสมอไป. ขอแนะนำให้ใช้ **แนวทางที่ 1 (Standalone)** ในการ debug ตรรกะภายใน `_process_data` จะง่ายและแน่นอนกว่า.
