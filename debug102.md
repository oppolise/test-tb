# วิธีการ Debug `loader.py` และ Child Processes ด้วย VS Code

เอกสารนี้อธิบายวิธีการ debug การทำงานของ `RunLoader` ในไฟล์ `loader.py` โดยตรง, รวมถึงการติดตาม (trace) การทำงานเข้าไปใน child processes ที่มันสร้างขึ้น. วิธีนี้จะทำให้คุณเห็นกระบวนการทำงานจริงของ profiler ตั้งแต่การรับไฟล์ trace ไปจนถึงการส่งต่อไปยัง parser ต่างๆ ใน `data.py`.

## ภาพรวมของแนวทาง

เราจะสร้างสคริปต์ Python (`debug_loader_direct_call.py`) ที่ทำหน้าที่เรียกใช้ `RunLoader` class โดยตรง. จากนั้น เราจะตั้งค่า VS Code Debugger (`launch.json`) ให้สามารถติดตามเข้าไปใน child processes ที่ `RunLoader` สร้างขึ้นได้. วิธีนี้จะทำให้เราสามารถตั้ง breakpoint ในเมธอด `_process_data` ซึ่งทำงานใน process แยก และไล่ดูการทำงานจริงได้.

## ขั้นตอนการ Debug

### 1. สร้างไฟล์สคริปต์ `debug_loader_direct_call.py`

สร้างไฟล์ใหม่ใน root directory ของโปรเจกต์ (ระดับเดียวกับ `tb_plugin/`) และตั้งชื่อว่า `debug_loader_direct_call.py`. ใส่โค้ดทั้งหมดด้านล่างนี้ลงไป:

```python
import os
import sys

# เพิ่ม path เพื่อให้สามารถ import tb_plugin ได้
# (อาจต้องปรับตามโครงสร้างโฟลเดอร์ของคุณ)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from tb_plugin.torch_tb_profiler.profiler.loader import RunLoader
from tb_plugin.torch_tb_profiler.io.cache import Cache
from tb_plugin.torch_tb_profiler.utils import get_logger

# ตั้งค่า logger เพื่อให้เห็น output ที่มีประโยชน์
logger = get_logger()

def main():
    """
    ฟังก์ชันหลักสำหรับสร้างและเรียกใช้ RunLoader
    """
    # 1. กำหนดไดเรกทอรีของ profiler run ที่มีหลาย worker
    # คุณสามารถเปลี่ยนเป็น "resnet50_ddp" หรือไดเรกทอรีอื่นได้
    run_dir_name = "resnet50_ddp_4_workers"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    run_dir = os.path.join(script_dir, 'tb_plugin/samples/', run_dir_name)

    if not os.path.isdir(run_dir):
        logger.error(f"Error: Directory not found at {run_dir}")
        logger.error("Please make sure you are running this script from the root of the kineto repository.")
        return

    logger.info(f"--- Preparing to debug RunLoader with directory: {run_dir_name} ---")

    # 2. สร้าง instance ของ Cache และ RunLoader
    # สามารถตั้ง breakpoint ที่นี่เพื่อตรวจสอบ object ที่เพิ่งสร้าง
    cache = Cache(os.path.join(script_dir, 'temp_cache'))
    run_loader = RunLoader(name=run_dir_name, run_dir=run_dir, caches=cache)

    logger.info("RunLoader instance created. Calling .load()...")
    # 3. เรียกเมธอด .load() ซึ่งเป็นจุดเริ่มต้นของกระบวนการทั้งหมด
    # debugger จะ step into เข้าไปในเมธอดนี้ในไฟล์ loader.py
    final_run_object = run_loader.load()

    # 4. ตรวจสอบผลลัพธ์สุดท้าย
    logger.info(f"\n--- RunLoader.load() finished ---")
    if final_run_object:
        logger.info(f"Total profiles loaded into final Run object: {len(final_run_object.profiles)}")
        # สามารถตั้ง breakpoint ที่นี่เพื่อสำรวจ final_run_object ที่สมบูรณ์ได้
    else:
        logger.warning("RunLoader.load() did not return a valid object.")

if __name__ == "__main__":
    main()

```

### 2. ตั้งค่า `launch.json` สำหรับ Debug Multiprocessing

นี่คือส่วนที่สำคัญที่สุด. เราต้องบอก VS Code ให้ติดตาม child processes.

*   ไปที่ Debug view (ไอคอนรูปแมลง), คลิกที่ไอคอนรูปเฟือง และเลือก **Python**.
*   เพิ่ม configuration ใหม่เข้าไปในไฟล์ `.vscode/launch.json`:

    ```json
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Debug RunLoader (Direct Call with Subprocess)",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/debug_loader_direct_call.py",
                "console": "integratedTerminal",
                "justMyCode": false,
                "subProcess": true
            }
        ]
    }
    ```
*   **คำอธิบาย Configuration**:
    *   `"name"`: ชื่อของ configuration นี้.
    *   `"request": "launch"`: บอกให้ debugger เริ่มรันโปรแกรมใหม่จากไฟล์ที่ระบุ.
    *   `"program"`: ระบุไฟล์ script ที่เราจะรัน.
    *   `"justMyCode": false`: **สำคัญมาก!** เพื่อให้ debugger สามารถหยุดในโค้ดของ library (`torch-tb_profiler`) ได้.
    *   `"subProcess": true"`: **หัวใจของการ debug multiprocessing.** Option นี้จะบอก `debugpy` (Python debugger ของ VS Code) ให้พยายาม attach debugger เข้าไปในทุกๆ child process ที่ถูกสร้างขึ้นโดย `RunLoader`.

### 3. การใช้งานและจุดที่น่าสนใจในการ Debug

1.  **ตั้ง Breakpoints**:
    *   **ใน `debug_loader_direct_call.py`**:
        *   ที่บรรทัด `run_loader = RunLoader(...)` เพื่อดูการสร้าง object.
        *   ที่บรรทัด `final_run_object = run_loader.load()` เพื่อเตรียม Step Into.
    *   **ใน `tb_plugin/torch_tb_profiler/profiler/loader.py`**:
        *   ในเมธอด `load()`, ตรงบรรทัดที่ `p.start()` เพื่อดูการสร้าง process.
        *   **ในเมธอด `_process_data()`, ที่บรรทัดแรกสุด.** **นี่คือ breakpoint ที่สำคัญที่สุด** สำหรับการ debug child process.
        *   ในเมธอด `_process_distributed_profiles()`, ตรง loop ที่ทำการ synchronize `real_time_ranges`.
    *   **ใน `tb_plugin/torch_tb_profiler/profiler/data.py`**:
        *   ในเมธอด `process()`, ที่บรรทัดแรกสุด เพื่อดูว่าข้อมูลถูกส่งต่อมาถึงที่นี่หรือไม่.

2.  **เริ่ม Debug**:
    *   ไปที่ Debug view, เลือก configuration **"Debug RunLoader (Direct Call with Subprocess)"** จาก drop-down list.
    *   กด **F5** (Start Debugging).

3.  **ติดตามการทำงาน**:
    *   Debugger จะหยุดที่ breakpoint แรกใน `debug_loader_direct_call.py`.
    *   กด **F11 (Step Into)** ที่บรรทัด `run_loader.load()` เพื่อเข้าไปในโค้ดของ `loader.py`.
    *   คุณสามารถไล่โค้ดในเมธอด `load()` ไปเรื่อยๆ.
    *   เมื่อโค้ดรันถึงบรรทัด `p.start()` ใน loop, `debugpy` จะเริ่มทำงานในเบื้องหลังเพื่อพยายาม attach กับ process ใหม่ที่กำลังจะถูกสร้าง.
    *   กด **F5 (Continue)** เพื่อให้ main process ทำงานต่อไปและสร้าง child processes.
    *   หลังจากนั้นไม่นาน, **debugger ควรจะหยุดที่ breakpoint ที่คุณตั้งไว้ใน `_process_data()`**.
    *   **ข้อสังเกต**: ในหน้าต่าง "CALL STACK" ของ VS Code, คุณอาจจะเห็นหลาย thread/process ปรากฏขึ้นมา. คุณสามารถสลับไปมาระหว่าง process เหล่านั้นเพื่อดูสถานะของแต่ละตัวได้. แต่ละ process จะหยุดที่ breakpoint ใน `_process_data()` ของตัวเอง.
    *   จากจุดนี้, คุณสามารถไล่ดูการทำงานของการ parse ไฟล์ trace แต่ละไฟล์, Step Into เข้าไปใน `RunProfileData.parse()`, และดูการทำงานของ parser อื่นๆ ได้อย่างสมบูรณ์.
    *   เมื่อ child process ทั้งหมดทำงานเสร็จ, debugger จะกลับมาหยุดที่ main process (ถ้ามี breakpoint เหลืออยู่) เพื่อให้คุณ debug ส่วน `_process_distributed_profiles` ต่อไปได้.

ด้วยวิธีนี้ คุณจะสามารถไล่ดูการทำงานของ `RunLoader` และการส่งต่อข้อมูลไปยัง `RunProfileData` ได้อย่างสมจริงและเป็นขั้นตอน.
