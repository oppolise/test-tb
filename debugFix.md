# วิธีแก้ไขปัญหาการ Debug เข้า Child Process ใน VS Code

เอกสารนี้อธิบายสาเหตุและวิธีการแก้ไขปัญหาที่ VS Code debugger ไม่สามารถ Step Into เข้าไปใน child process ที่สร้างโดย `multiprocessing.Process` ได้ แม้ว่าจะตั้งค่า `"subProcess": true` ใน `launch.json` แล้วก็ตาม.

## สาเหตุของปัญหา

สาเหตุหลักที่ทำให้ debugger ไม่สามารถติดตามเข้าไปใน child process ได้อย่างสม่ำเสมอ มาจาก **"Start Method"** ของ `multiprocessing` ใน Python.

*   `multiprocessing` มี Start Method หลายแบบ เช่น `fork`, `spawn`.
*   **บน Linux/macOS**: ค่าเริ่มต้นคือ `'fork'`. การ `fork` จะคัดลอก memory space ของ parent process ไปยัง child process เกือบทั้งหมด. วิธีนี้รวดเร็ว แต่สถานะของ debugger (debugpy) ที่ทำงานอยู่ใน parent process อาจไม่ถูกส่งต่อไปยัง child process อย่างสมบูรณ์ ทำให้ debugger "หลุด" และไม่สามารถติดตามต่อได้.
*   **บน Windows/macOS (ใหม่)**: ค่าเริ่มต้นคือ `'spawn'`. การ `spawn` จะสร้าง process ใหม่ที่ "สะอาด" (clean state) แล้วค่อย import script และรันฟังก์ชันเป้าหมาย. **`debugpy` (debugger ของ VS Code) ทำงานได้ดีและน่าเชื่อถือที่สุดกับวิธีนี้** เพราะมันสามารถ inject ตัวเองเข้าไปใน process ใหม่ที่สะอาดได้ง่ายกว่า.

ดังนั้น ปัญหาที่คุณเจอคือ มีความเป็นไปได้สูงที่สภาพแวดล้อมของคุณกำลังใช้ start method แบบ `'fork'` ซึ่งทำให้ `debugpy` ทำงานผิดพลาด.

## วิธีการแก้ไข (แนะนำเป็นอย่างยิ่ง)

วิธีแก้ที่ตรงจุดและมีประสิทธิภาพที่สุดคือการ **บังคับให้ `multiprocessing` ใช้ start method แบบ `'spawn'`** ในสคริปต์ debug ของเรา.

1.  **เปิดไฟล์สคริปต์ Debug ของคุณ**: เปิดไฟล์ `debug_loader_direct_call.py` (จาก `debug102.md`).

2.  **แก้ไขส่วนท้ายของไฟล์**: แก้ไขส่วน `if __name__ == "__main__":` ให้เป็นดังนี้:

    ```python
    import multiprocessing as mp

    # ... (โค้ดส่วนฟังก์ชัน main() เหมือนเดิม) ...


    if __name__ == "__main__":
        # --- เพิ่มโค้ดส่วนนี้เข้าไป ---
        # บังคับให้ multiprocessing ใช้ start method แบบ 'spawn'
        # ซึ่งทำงานกับ debugger ได้ดีกว่า 'fork'
        # ต้องวางโค้ดนี้ไว้ใน if __name__ == "__main__": block เท่านั้น
        # เพื่อป้องกันการเกิด loop ไม่รู้จบใน process ที่ spawn ใหม่
        # force=True ใช้ในกรณีที่ start method อาจถูกตั้งค่าไปแล้ว
        try:
            mp.set_start_method('spawn', force=True)
            print("--- Multiprocessing start method set to 'spawn' ---")
        except RuntimeError:
            # อาจเกิด RuntimeError ถ้า context ถูกตั้งค่าไปแล้วและ force=False
            # แต่โดยทั่วไปในสคริปต์นี้จะทำงานได้ปกติ
            print("--- Multiprocessing start method was already set ---")

        # เรียกใช้ฟังก์ชัน main ของคุณ
        main()
    ```

3.  **บันทึกไฟล์**.

### การทำงานของโค้ดที่เพิ่มเข้าไป

*   `import multiprocessing as mp`: import library ที่จำเป็น.
*   `mp.set_start_method('spawn', force=True)`: เป็นการสั่งให้ Python ใช้ `'spawn'` เป็นวิธีในการสร้าง process ใหม่ทั้งหมดที่เกิดขึ้นหลังจากนี้.
*   `if __name__ == "__main__":`: **สำคัญมาก!** โค้ดที่เรียก `set_start_method` และโค้ดหลักที่จะรันต้องอยู่ภายใน block นี้เสมอ. เพราะเมื่อ child process ถูก `spawn` ขึ้นมา, มันจะ import สคริปต์ทั้งหมดใหม่ตั้งแต่ต้น. การมีเงื่อนไขนี้จะป้องกันไม่ให้ child process รันโค้ดส่วนนี้ซ้ำอีก ซึ่งจะทำให้เกิด loop การสร้าง process ไม่รู้จบ.

## ตรวจสอบการตั้งค่า `launch.json` อีกครั้ง

เพื่อให้แน่ใจว่าทุกอย่างถูกต้อง, ให้ตรวจสอบไฟล์ `.vscode/launch.json` ของคุณว่ามีค่าที่สำคัญครบถ้วน:

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
*   `"justMyCode": false`: ต้องเป็น false.
*   `"subProcess": true`: ต้องเป็น true.
*   `"console": "integratedTerminal"`: แนะนำให้ใช้ค่านี้.

## ทดสอบการ Debug อีกครั้ง

หลังจากแก้ไขสคริปต์ `debug_loader_direct_call.py` แล้ว ให้ทำตามขั้นตอนการ debug เดิม:

1.  **ตั้ง Breakpoint** ใน `loader.py` ที่บรรทัดแรกของเมธอด `_process_data`.
2.  **เริ่ม Debug** โดยใช้ configuration "Debug RunLoader (Direct Call with Subprocess)".
3.  เมื่อ debugger หยุดที่ breakpoint ใน `main()` หรือใน `load()`, กด **F5 (Continue)**.
4.  ตอนนี้ debugger ควรจะสามารถหยุดที่ breakpoint ที่คุณตั้งไว้ใน `_process_data` ได้อย่างถูกต้อง เพราะ child process ถูกสร้างขึ้นมาด้วยวิธีที่ debugger รองรับอย่างสมบูรณ์.

ด้วยการแก้ไขนี้ คุณจะสามารถไล่ดูการทำงานของโค้ดที่รันใน child process ได้อย่างที่ต้องการ.
