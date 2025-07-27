# คำอธิบายโครงสร้างไฟล์ `pt.trace.json` เชิงลึก

เอกสารนี้อธิบายโครงสร้างและรายละเอียดของ key-value ต่างๆ ที่พบในไฟล์ `*.pt.trace.json.gz` ซึ่งเป็นผลลัพธ์จาก `torch.profiler` โดยเจาะลึกไปที่แต่ละ key และ value ที่คุณสนใจ.

ไฟล์นี้ใช้ [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview) เป็นพื้นฐาน.

## 1. โครงสร้างไฟล์ระดับบนสุด (Top-level Keys)

เป็น keys ที่ให้ข้อมูลภาพรวม (metadata) ของการ profile.

*   **`traceName`**: `str`
    *   **คืออะไร**: ชื่อของ trace ทั้งหมด. ในทางปฏิบัติ, `torch-tb-profiler` ไม่ได้ใช้ key นี้โดยตรง แต่จะใช้ชื่อของไดเรกทอรี run แทน.
*   **`displayTimeUnit`**: `str`
    *   **คืออะไร**: หน่วยเวลาที่แนะนำให้ UI แสดงผล. ค่าที่เป็นไปได้คือ `"ns"` (nanoseconds) หรือ `"ms"` (microseconds).
    *   **หมายเหตุ**: แม้จะมี key นี้, แต่ค่า `ts` และ `dur` ใน trace ของ PyTorch จะเป็น **ไมโครวินาที (microseconds)** เสมอ.
*   **`baseTimeNanoseconds`**: `int`
    *   **คืออะไร**: เวลาอ้างอิง (epoch time) ในหน่วยนาโนวินาที.
    *   **หมายเหตุ**: `torch-tb-profiler` ไม่ได้ใช้ key นี้ในการคำนวณ เพราะค่า `ts` ในแต่ละ event เป็น absolute timestamp อยู่แล้ว.

---

## 2. ความแตกต่างระหว่าง `ts` และ `dur`

*   **`"ts"` (Timestamp)**
    *   **คืออะไร**: **จุดเวลาที่ event เริ่มต้นขึ้น (start time)**.
    *   **หน่วย**: ไมโครวินาที (microseconds).
    *   **การใช้งาน**: บอกว่า event เริ่มขึ้น "เมื่อไหร่" บนไทม์ไลน์.

*   **`"dur"` (Duration)**
    *   **คืออะไร**: **ระยะเวลาที่ event นั้นใช้ไป**.
    *   **หน่วย**: ไมโครวินาที (microseconds).
    *   **การใช้งาน**: บอกว่า event ทำงาน "นานเท่าไหร่". ใช้คู่กับ `ts` เพื่อคำนวณเวลาสิ้นสุด (`end_time = ts + dur`). Key นี้จะปรากฏเฉพาะใน event ที่มี `"ph": "X"`.

---

## 3. โครงสร้างภายใน `traceEvents`

แต่ละ object ใน list `traceEvents` จะมี key-value ที่อธิบายกิจกรรม (event) หนึ่งๆ.

### 3.1 `"cat"` (Category)

*   **คืออะไร**: เป็น key ที่ใช้จัดหมวดหมู่ของ event. ค่าของ `cat` จะถูก map ไปยัง `EventTypes` ที่เป็นมาตรฐานภายในโดย `EventTypeMap` ใน `trace.py`.
*   **Value ที่เป็นไปได้และความหมาย**:
    *   `"python_function"`: การเรียกใช้ Python function ทั่วไป.
    *   `"cpu_op"`: CPU-side operator ที่ถูกบันทึกโดย `record_function` ของ PyTorch.
    *   `"cpu_instant_event"`: event ที่เกิดขึ้นทันทีบน CPU.
    *   `"ac2g"`: (Async CPU to GPU) event ที่แสดงถึงการส่งต่องานจาก CPU ไปยัง GPU.
    *   `"cuda_runtime"`: การเรียกใช้ CUDA Runtime API call จากฝั่ง CPU (เช่น `cudaLaunchKernel`).
    *   `"kernel"`: การทำงานของ GPU kernel จริงๆ บน device.
    *   `"fwdbwd"`: category พิเศษสำหรับ "flow" events (`ph: 's'`, `ph: 'f'`) ที่ใช้ในการสร้างความสัมพันธ์ forward-backward.
    *   `"gpu_memcpy"`: การคัดลอกข้อมูล (memory copy) บน GPU.
    *   `"cuda_driver"`: การเรียกใช้ CUDA Driver API (low-level).
    *   `"gpu_memset"`: การตั้งค่าพื้นที่ในหน่วยความจำ (memory set) บน GPU.
    *   `"user_annotation"`, `"gpu_user_annotation"`: event ที่เกิดจากการที่ผู้ใช้ใส่ `record_function("...")` หรือ `nvtx.range_push("...")` เข้าไปในโค้ดเอง.
    *   `"Trace"`: event ที่ให้ข้อมูลเกี่ยวกับตัว profiler เอง เช่น `PyTorch Profiler (0)`.

### 3.2 `"pid"` (Process ID)

*   **คืออะไร**: หมายเลข Process ID ของ OS. แต่ใน PyTorch Profiler trace, ความหมายของมันจะเปลี่ยนไปตามบริบท.
*   **Value ที่เป็นไปได้และความหมาย**:
    *   **เลข ID (เช่น `1352521`)**: สำหรับ CPU events, นี่คือ Process ID จริงๆ ของ Python process.
    *   **GPU ID (เช่น `0`, `1`, `2`, `3`)**: สำหรับ GPU events (`kernel`, `gpu_memcpy`), `pid` จะถูกใช้เพื่อระบุว่า event นี้เกิดขึ้นบน **GPU device ใด**.
    *   **`"Spans"`**: `pid` สังเคราะห์ที่ใช้ใน UI เพื่อจัดกลุ่ม profiler steps (spans) ทั้งหมดไว้ใน lane (แถว) เดียวกัน.
    *   **`"Traces"`**: `pid` สังเคราะห์ที่ใช้ใน UI เพื่อจัดกลุ่ม metadata ของ trace ทั้งหมดไว้ใน lane เดียวกัน.

### 3.3 `"tid"` (Thread ID)

*   **คืออะไร**: หมายเลข Thread ID ของ OS.
*   **Value ที่เป็นไปได้และความหมาย**:
    *   **เลข ID (เช่น `1359383`)**: สำหรับ CPU events, นี่คือ Thread ID จริงๆ ที่รัน operator นั้น.
    *   **CPU Thread ที่สั่งงาน GPU**: สำหรับ GPU events, `tid` ไม่ได้หมายถึง thread บน GPU แต่หมายถึง **CPU Thread ID ที่ทำการ launch** kernel นั้น.
    *   **`"PyTorch Profiler"`**: `tid` สังเคราะห์ที่ใช้ใน UI สำหรับ lane ที่แสดง profiler steps.
    *   **`"Trace PyTorch Profiler"`**: `tid` สังเคราะห์ที่ใช้ใน UI สำหรับ lane ที่แสดง metadata ของ trace.

### 3.4 `"name"`

*   **คืออะไร**: ชื่อของ event ที่มนุษย์อ่านเข้าใจได้.
*   **Value ที่เป็นไปได้และความหมาย**:
    *   `"[memory]"`: ชื่อเฉพาะสำหรับ memory event (`ph: 'i'`).
    *   `"ac2g"`: ชื่อสำหรับ event ที่ส่งต่องานจาก CPU ไป GPU.
    *   `"cudaLaunchKernel"`: ชื่อของ CUDA Runtime API call.
    *   **ชื่อ Operator (เช่น `"aten::add"`)**: ชื่อของ PyTorch operator.
    *   **ชื่อ Kernel (demangled)**: ชื่อของ GPU kernel ที่ผ่านการ demangle แล้ว (ทำให้อ่านง่ายขึ้น).

### 3.5 `"id"`

*   **คืออะไร**: เป็น ID ที่ใช้สำหรับสร้างความสัมพันธ์ระหว่าง event ที่ไม่ต่อเนื่องกัน.
*   **ความสำคัญ**: มีบทบาทสำคัญอย่างยิ่งใน "flow" events (`ph: 's'`, `ph: 'f'`). event ที่เป็นจุดเริ่มต้นและจุดสิ้นสุดของ flow เดียวกัน (เช่น forward และ backward op ที่คู่กัน) จะมี **`id` เดียวกัน**.

### 3.6 `"s"` (Scope)

*   **คืออะไร**: เป็น key ที่ใช้ใน instant events (`ph: 'i'`) เพื่อบอก scope ของ event นั้น.
*   **Value ที่เป็นไปได้และความหมาย**:
    *   `"t"`: **Thread scope**. event นี้มีความหมายในระดับ thread.
    *   `"g"`: **Global scope**. event นี้มีความหมายในระดับ global ของ trace ทั้งหมด.
    *   `"p"`: **Process scope**. event นี้มีความหมายในระดับ process.

### 3.7 `"args"`

*   **คืออะไร**: เป็น object (dictionary) ที่เก็บข้อมูลเพิ่มเติมของ event.
*   **Value ภายในที่สำคัญและความหมาย**:
    *   `"Addr"`: Address ของหน่วยความจำ (ใน memory events).
    *   `"Bytes"`: ขนาดของหน่วยความจำ (เป็น byte) ที่ถูกจัดสรร (ค่าบวก) หรือคืนค่า (ค่าลบ).
    *   `"Collective name"`, `"In msg nelems"`, `"Group size"`, `"Process Group Ranks"`: arguments ที่เกี่ยวข้องกับ communication operations (เช่น AllReduce).
    *   `"Input Dims"`, `"Input type"`: ข้อมูลเกี่ยวกับ input tensor ของ operator (ขนาด, ประเภทข้อมูล).
    *   `"Device Id"`, `"Device Type"`: ID และประเภทของ device ที่ memory event เกิดขึ้น.
    *   `"External id"`: ID ที่ใช้เชื่อมโยง CPU operator กับ GPU kernel/runtime ที่มัน launch. **สำคัญมากสำหรับ `EventParser`**.
    *   `"Fwd thread id"`: ID ของ forward thread ที่เกี่ยวข้องกับ backward op นี้.
    *   `"Python id"`, `"Python module id"`, `"Python parent id"`: ID ที่ใช้สร้าง call stack ของ Python.
    *   `"Record function id"`, `"Sequence number"`: ID ภายในของ `record_function` ของ PyTorch.
    *   `"Total Allocated"`, `"Total Reserved"`: ขนาดหน่วยความจำทั้งหมดที่ถูกจัดสรร/จอง ณ เวลาที่ memory event เกิดขึ้น.
    *   `"block"`, `"grid"`: ขนาดของ grid และ block สำหรับ GPU kernel.
    *   `"blocks per sm"`, `"est. achieved occupancy %"`, `"registers per thread"`, `"shared memory"`: performance metrics ของ GPU kernel.
    *   `"bytes"`: (ใน kernel args) จำนวน byte ที่ถูกประมวลผล.
    *   `"cbid"`: Callback ID ของ CUPTI.
    *   `"correlation"`: ID ที่ใช้เชื่อมโยง CUDA runtime call กับ GPU kernel/memcpy ที่มันเรียก. **สำคัญมากสำหรับ `EventParser`**.
    *   `"device"`: ID ของ GPU ที่ kernel ทำงาน.
    *   `"stream"`: CUDA stream ID ที่ event นี้ทำงาน.
    *   `"labels"`: label ที่ผู้ใช้กำหนดเอง.
    *   `"name"`: (ใน args) ชื่อเพิ่มเติม.
    *   `"sort_index"`: index สำหรับช่วยเรียงลำดับใน UI.
