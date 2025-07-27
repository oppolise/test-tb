# คำอธิบายโครงสร้างไฟล์ `pt.trace.json` ของ PyTorch Profiler

เอกสารนี้อธิบายโครงสร้างและรายละเอียดของ key-value ต่างๆ ที่พบในไฟล์ `*.pt.trace.json.gz` ซึ่งเป็นผลลัพธ์จาก `torch.profiler` โดยอิงตามข้อมูลตัวอย่างและโค้ดใน `torch-tb-profiler`.

ไฟล์นี้ใช้ [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview) เป็นพื้นฐาน.

## 1. โครงสร้างไฟล์ระดับบนสุด (Top-level Keys)

เป็น keys ที่ให้ข้อมูลภาพรวม (metadata) ของการ profile ครั้งนั้นๆ.

*   **`schemaVersion`**: `int`
    *   **คืออะไร**: เวอร์ชันของ schema ของไฟล์ trace. ช่วยให้ profiler plugin รู้ว่าควรจะแปลความหมายของข้อมูลในไฟล์นี้อย่างไร.
    *   **โค้ดที่เกี่ยวข้อง**: `RunProfileData.data_schema_version`.

*   **`deviceProperties`**: `List[object]`
    *   **คืออะไร**: List ของ object, โดยแต่ละ object จะมีข้อมูลคุณสมบัติทางกายภาพของ GPU แต่ละตัวที่ใช้ในการรัน.
    *   **Value ภายใน (ต่อ 1 GPU)**:
        *   `"id"`: ID ของ GPU (เช่น 0, 1, 2, 3).
        *   `"name"`: ชื่อรุ่นของ GPU (เช่น "NVIDIA A100-SXM4-40GB").
        *   `"totalGlobalMem"`: ขนาดหน่วยความจำทั้งหมดของ GPU (เป็น bytes).
        *   `"computeMajor"`, `"computeMinor"`: Compute Capability version ของ GPU (เช่น 8.0). ใช้ในการตรวจสอบว่า GPU รองรับฟีเจอร์บางอย่างหรือไม่ (เช่น Tensor Cores).
        *   `"numSms"`: จำนวน Streaming Multiprocessors (SMs) บน GPU.
        *   และคุณสมบัติอื่นๆ ของ CUDA device เช่น `maxThreadsPerBlock`, `regsPerBlock`, `warpSize`, `sharedMemPerBlock`.
    *   **โค้ดที่เกี่ยวข้อง**: `RunProfileData.device_props`.

*   **`distributedInfo`**: `object`
    *   **คืออะไร**: object ที่มีข้อมูลเกี่ยวกับการตั้งค่า distributed training.
    *   **Value ภายใน**:
        *   `"backend"`: backend ที่ใช้สำหรับการสื่อสาร (เช่น "nccl").
        *   `"rank"`: rank ของ worker ที่สร้างไฟล์ trace นี้.
        *   `"world_size"`: จำนวน process ทั้งหมดใน distributed job.
        *   `"pg_config"`: list ของการตั้งค่า Process Group.
        *   `"nccl_version"`: เวอร์ชันของ NCCL ที่ใช้.
    *   **โค้ดที่เกี่ยวข้อง**: `RunProfileData.distributed_info`.

*   **`cupti_version`**, **`cuda_runtime_version`**, **`cuda_driver_version`**: `int`
    *   **คืออะไร**: หมายเลขเวอร์ชันของส่วนประกอบต่างๆ ของ CUDA ที่ใช้ในการรัน (CUPTI, CUDA Runtime, และ CUDA Driver).

*   **`record_shapes`**, **`profile_memory`**, **`with_stack`**: `int` (0 หรือ 1)
    *   **คืออะไร**: เป็น flags ที่บอกว่าตอนที่รัน `torch.profiler` ได้มีการเปิดใช้งาน option เหล่านี้หรือไม่. `1` คือเปิด, `0` คือปิด.
        *   `record_shapes=1`: บันทึกขนาด (shape) ของ tensor ที่เป็น input ของ operator.
        *   `profile_memory=1`: เปิดใช้งานการ profile memory.
        *   `with_stack=1`: บันทึก call stack สำหรับ operator.

*   **`trace_id`**: `str`
    *   **คืออะไร**: ID ที่ไม่ซ้ำกันสำหรับ trace run นี้.

*   **`traceEvents`**: `List[object]`
    *   **คืออะไร**: **ส่วนที่สำคัญที่สุดของไฟล์**. เป็น list ของ event ทั้งหมดที่เกิดขึ้นระหว่างการ profile. แต่ละ event จะเป็น object (dictionary) ที่มีรายละเอียดของตัวเอง.

---

## 2. โครงสร้างภายใน `traceEvents`

แต่ละ object ใน list `traceEvents` จะมี key-value ที่อธิบายกิจกรรม (event) หนึ่งๆ.

### 2.1 Keys พื้นฐานในทุก Event

*   **`"name"`**: `str`
    *   **คืออะไร**: ชื่อของ event ที่มนุษย์อ่านเข้าใจได้ เช่น ชื่อของ operator (`aten::add`), ชื่อของ kernel, หรือชื่อของ annotation ที่ผู้ใช้กำหนดเอง.

*   **`"pid"` (Process ID)**: `int`
    *   **คืออะไร**: หมายเลข Process ID ของ OS ที่ event นี้เกิดขึ้น.
    *   **ความหมายพิเศษ**: ในบริบทของ CUDA events, `pid` มักจะถูกใช้เพื่อระบุ **GPU ID** ที่ kernel หรือ memory copy นั้นทำงาน.

*   **`"tid"` (Thread ID)**: `int`
    *   **คืออะไร**: หมายเลข Thread ID ของ OS ที่ event นี้เกิดขึ้น.
    *   **ต่างอะไรกับ `"pid"`**: `pid` คือ Process ID ซึ่งเป็นหน่วยที่ใหญ่กว่า. หนึ่ง process สามารถมีได้หลาย thread. สำหรับ CPU events, `tid` คือ thread ที่รัน operator นั้น. สำหรับ GPU events, `tid` คือ **CPU thread ที่ทำการ launch** kernel นั้น.

*   **`"ts"` (Timestamp)**: `float` หรือ `int`
    *   **คืออะไร**: เวลาที่ event เริ่มต้นขึ้น. หน่วยเป็น **ไมโครวินาที (microseconds)**.

### 2.2 `"ph"` (Phase) - ประเภทของ Event

`"ph"` เป็น key ที่สำคัญที่สุด ใช้เพื่อกำหนด "ลักษณะ" ของ event ในไทม์ไลน์.

| ค่า `ph` | ชื่อเต็ม | ความหมาย | Keys ที่ต้องมี | ถูกใช้ใน Profiler สำหรับ |
| :---: | :--- | :--- | :--- | :--- |
| **`X`** | Complete Event | event ที่มีจุดเริ่มต้นและระยะเวลาที่ชัดเจน | `ts`, `dur` | CPU ops, GPU kernels, Memcpy/Memset (ส่วนใหญ่ของ trace) |
| **`i`** | Instant Event | event ที่เกิดขึ้น ณ จุดเวลาเดียว ไม่มีระยะเวลา | `ts` | Memory snapshots (`name: "[memory]"`) |
| **`s`** | Flow Start | จุดเริ่มต้นของ "flow" หรือความสัมพันธ์ | `id` | **Forward/Backward association** (จุดเริ่มต้นของ forward op) |
| **`f`** | Flow Finish | จุดสิ้นสุดของ "flow" หรือความสัมพันธ์ | `id` | **Forward/Backward association** (จุดที่ backward op คู่กันเกิดขึ้น) |
| **`C`** | Counter Event | บันทึกค่าของตัวนับ (counter) ที่เปลี่ยนไปตามเวลา | `ts`, `args` | สร้างกราฟ GPU Utilization, SM Efficiency (สร้างขึ้นทีหลัง ไม่ได้มาจาก trace ดั้งเดิม) |
| **`M`**` | Metadata Event | ให้ข้อมูล metadata เกี่ยวกับ process หรือ thread | `args` | ตั้งชื่อ Process (เช่น GPU 0) หรือ Thread ใน UI |

### 2.3 `"cat"` (Category) - หมวดหมู่ของ Event

`"cat"` เป็น key ที่ profiler ใช้เพื่อจัดหมวดหมู่ของ event. ค่าเหล่านี้จะถูก map ไปยัง `EventTypes` ที่เป็นมาตรฐานภายในโดยใช้ `EventTypeMap` ใน `trace.py`.

| ค่า `cat` (ตัวอย่าง) | ถูก Map ไปยัง `EventType` | สร้างเป็น Object Class | ความหมาย |
| :--- | :--- | :--- | :--- |
| **`cpu_op`**, **`operator`** | `OPERATOR` | `OperatorEvent` | CPU-side operator ที่ถูกบันทึกโดย `record_function` ของ PyTorch. |
| **`kernel`** | `KERNEL` | `KernelEvent` | GPU kernel ที่ถูก launch และบันทึกผ่าน CUPTI. |
| **`memcpy`**, **`gpu_memcpy`** | `MEMCPY` | `DurationEvent` | การคัดลอกข้อมูล (memory copy) เช่น H2D, D2H, D2D. |
| **`memset`**, **`gpu_memset`** | `MEMSET` | `DurationEvent` | การตั้งค่าพื้นที่ในหน่วยความจำ (memory set). |
| **`runtime`** | `RUNTIME` | `DurationEvent` | การเรียกใช้ CUDA Runtime API call จากฝั่ง CPU (เช่น `cudaLaunchKernel`). |
| **`python_function`** | `PYTHON_FUNCTION`| `PythonFunctionEvent` | การเรียกใช้ Python function ที่ถูกติดตามโดย profiler. |
| **`user_annotation`** | `USER_ANNOTATION`| `OperatorEvent` | Event ที่ผู้ใช้ใส่ `record_function("...")` เข้าไปในโค้ดเอง. |
| **`memory`** | `MEMORY` | `MemoryEvent` | Instant event ที่บันทึกสถานะการจัดสรร/คืนหน่วยความจำ. |
| **`fwdbwd`** | (ไม่ถูก map) | (ไม่สร้าง object) | Category พิเศษสำหรับ flow events (`ph: 's'`, `ph: 'f'`) ที่ใช้ในการสร้างความสัมพันธ์ forward-backward. |

### 2.4 Keys เพิ่มเติมและ `args`

*   **`"dur"` (Duration)**: `float` หรือ `int`
    *   **คืออะไร**: ระยะเวลาที่ event นั้นใช้. หน่วยเป็น **ไมโครวินาที (microseconds)**. Key นี้จะปรากฏใน event ที่มี `"ph": "X"` เท่านั้น.

*   **`"args"`**: `object`
    *   **คืออะไร**: object (dictionary) ที่เก็บข้อมูลเพิ่มเติมของ event นั้นๆ ซึ่งจะแตกต่างกันไปตามประเภทของ event.
    *   **ตัวอย่าง Value ภายในที่สำคัญ**:
        *   `"External id"`: ID ที่ใช้เชื่อมโยง CPU operator กับ GPU kernel/runtime ที่มัน launch.
        *   `"correlation"`: ID ที่ใช้เชื่อมโยง CUDA runtime call กับ GPU kernel/memcpy.
        *   `"Input Dims"` / `"Input dims"`: ขนาด (shape) ของ input tensors.
        *   `"Call stack"`: call stack ของ operator.
        *   `"Device Id"`, `"Bytes"`, `"Addr"`: ข้อมูลสำหรับ memory events.
        *   `"grid"`, `"block"`, `"est. achieved occupancy %"`: ข้อมูลสำหรับ kernel events.
        *   `"Fwd thread id"`: ID ของ forward thread ที่เกี่ยวข้องกับ backward op นี้.
