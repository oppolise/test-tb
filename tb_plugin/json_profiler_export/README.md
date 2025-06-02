# JSON Profiler Export Plugin (ปลั๊กอินสำหรับส่งออกข้อมูล Profiler เป็น JSON)

## ภาพรวม (Overview)

ปลั๊กอินนี้เป็นส่วนเสริมสำหรับ TensorBoard ที่ช่วยให้ผู้ใช้งานสามารถดึงข้อมูลการโปรไฟล์ PyTorch (PyTorch Profiler) ที่ถูกบันทึกไว้ ออกมาเป็นไฟล์ JSON ไฟล์เดียว ไฟล์ JSON นี้จะรวมข้อมูลสรุปต่างๆ เช่น ภาพรวมการทำงาน, สถิติของ Operators, สถิติของ Kernels, ข้อมูลการใช้หน่วยความจำ และคำแนะนำต่างๆ ที่ได้จากการวิเคราะห์ข้อมูลโปรไฟล์ ทำให้สะดวกต่อการนำข้อมูลไปวิเคราะห์ต่อหรือจัดเก็บ

## ส่วนประกอบหลัก (Main Components)

ปลั๊กอินนี้ประกอบด้วยสองส่วนหลัก:

1.  **`plugin.py`**:
    *   ทำหน้าที่เป็นจุดเชื่อมต่อกับ TensorBoard
    *   ลงทะเบียนปลั๊กอินกับ TensorBoard ทำให้สามารถเข้าถึงได้ผ่านหน้าเว็บ UI
    *   จัดการคำขอ HTTP (HTTP requests) ที่เข้ามายังปลั๊กอิน
    *   มีหน้า HTML แบบง่าย (`static/index.html`) สำหรับให้ผู้ใช้กดดาวน์โหลดไฟล์ JSON
    *   เมื่อมีการร้องขอการดาวน์โหลด จะเรียกใช้โมดูล `run_export.py` เพื่อประมวลผลข้อมูล

2.  **`run_export.py`**:
    *   เป็นส่วนที่ทำการประมวลผลข้อมูลโปรไฟล์จริงๆ
    *   สแกนหาไฟล์ trace ของ PyTorch Profiler (เช่น `*.pt.trace.json` หรือ `*.pt.trace.json.gz`) ที่อยู่ใน `logdir` ที่ TensorBoard กำลังอ่านอยู่
    *   ใช้ไลบรารี `torch_tb_profiler` (บางส่วน) เพื่อแยกวิเคราะห์ (parse) และประมวลผลไฟล์ trace เหล่านั้น
    *   สร้างออบเจ็กต์ `RunProfile` ซึ่งเก็บข้อมูลที่ผ่านการประมวลผลแล้ว เช่น ภาพรวม (overview), สถิติเกี่ยวกับ operations, kernels, การใช้หน่วยความจำ และคำแนะนำต่างๆ
    *   รวบรวมข้อมูลทั้งหมดนี้ให้อยู่ในรูปแบบ Dictionary ของ Python เพื่อเตรียมส่งออกเป็น JSON

## การติดตั้งและใช้งาน (Installation and Usage)

### การติดตั้ง (Installation)

1.  **ตรวจสอบว่ามี `torch_tb_profiler`**: ปลั๊กอินนี้ใช้โค้ดบางส่วนจาก `torch_tb_profiler` ดังนั้นคุณควรมี `torch-tb-plugin` ติดตั้งอยู่ในสภาพแวดล้อม Python ของคุณ (ปกติจะมาพร้อมกับ PyTorch เวอร์ชันใหม่ๆ หรือติดตั้งแยกได้)
2.  **ตำแหน่งของปลั๊กอิน**:
    *   นำโฟลเดอร์ `json_profiler_export` ทั้งหมดไปวางไว้ในไดเรกทอรี `tb_plugin` ซึ่งเป็นที่ TensorBoard (และ `torch_tb_profiler` ดั้งเดิม) ใช้ค้นหาปลั๊กอิน โดยทั่วไป หากคุณติดตั้ง `torch-tb-plugin` ผ่าน pip ตำแหน่ง `tb_plugin` อาจจะอยู่ใน `site-packages` ของ Python environment ของคุณ หรือคุณอาจจะต้องสร้างโครงสร้างโฟลเดอร์นี้ขึ้นมาเองหากต้องการพัฒนาปลั๊กอินแบบ local
    *   โครงสร้างที่คาดหวัง:
        ```
        <python_site_packages_or_project_root>/
        └── tb_plugin/
            ├── torch_tb_profiler/
            │   └── (ไฟล์ต่างๆ ของ torch_tb_profiler)
            └── json_profiler_export/
                ├── __init__.py
                ├── plugin.py
                ├── run_export.py
                ├── static/
                │   └── index.html
                └── README.md
        ```

### การใช้งาน (Usage)

1.  **รัน PyTorch Profiler**: สร้างข้อมูลโปรไฟล์จากโมเดล PyTorch ของคุณโดยใช้ `torch.profiler` และบันทึกผลลัพธ์ลงในไดเรกทอรี (log directory) ตัวอย่างเช่น:
    ```python
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/my_profile_run'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        # Your model training/inference steps
        for step, batch_data in enumerate(data_loader):
            if step >= (1 + 1 + 3) * 2: # (wait + warmup + active) * repeat
                break
            model(batch_data)
            prof.step()
    ```
    ข้อมูลโปรไฟล์จะถูกเก็บไว้ใน `./logs/my_profile_run` (หรือตามที่คุณตั้งค่า)

2.  **รัน TensorBoard**:
    *   เปิด Terminal หรือ Command Prompt
    *   สั่งรัน TensorBoard โดยชี้ `logdir` ไปยังไดเรกทอรีที่คุณบันทึกข้อมูลโปรไฟล์ไว้ ตัวอย่าง:
        ```bash
        tensorboard --logdir ./logs/my_profile_run
        ```
    *   หากคุณมีหลาย run หรือต้องการให้ TensorBoard สแกนไดเรกทอรีแม่:
        ```bash
        tensorboard --logdir ./logs
        ```

3.  **เข้าถึงปลั๊กอิน**:
    *   เปิดเว็บเบราว์เซอร์แล้วไปที่ URL ที่ TensorBoard แสดง (ปกติคือ `http://localhost:6006/`)
    *   ในหน้า TensorBoard UI มองหาแท็บหรือเมนูดรอปดาวน์ที่ชื่อว่า "JSON Export Profiler" (หรือชื่อ `plugin_name` ที่ตั้งไว้ใน `plugin.py`)
    *   คลิกเข้าไปยังหน้าของปลั๊กอิน

4.  **ดาวน์โหลดไฟล์ JSON**:
    *   ในหน้าของปลั๊กอิน "JSON Export Profiler" คุณจะเห็นคำอธิบายและปุ่ม "Download Profiler Data as JSON"
    *   คลิกที่ปุ่มนี้เพื่อดาวน์โหลดไฟล์ `profiler_export.json` ซึ่งจะบรรจุข้อมูลโปรไฟล์ที่ประมวลผลแล้ว

## โครงสร้างไฟล์ JSON ที่ส่งออก (Output JSON Structure)

ไฟล์ `profiler_export.json` ที่ได้จะมีโครงสร้างข้อมูลที่ประกอบด้วย:
*   `workers`: dictionary ที่เก็บข้อมูลของแต่ละ worker (trace file) ที่ถูกประมวลผล
    *   แต่ละ worker จะมีข้อมูลเช่น `overview`, `recommendations`, `operation_pie_by_name`, `kernel_pie`, `memory_summary`, `memory_curve` และอื่นๆ
*   `all_recommendations`: รายการคำแนะนำทั้งหมดที่รวบรวมจากทุก workers (ตัดรายการซ้ำซ้อนออก)
*   `errors`: รายการข้อผิดพลาดที่อาจเกิดขึ้นระหว่างการประมวลผลไฟล์

ข้อมูลนี้สามารถนำไปใช้ในการวิเคราะห์ประสิทธิภาพของโมเดล PyTorch ของคุณ หรือสร้างการแสดงผล (visualizations) แบบกำหนดเองได้
