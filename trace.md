# อธิบายโค้ด `trace.py` เชิงลึก

เอกสารนี้จะอธิบายการทำงานของไฟล์ `tb_plugin/torch_tb_profiler/profiler/trace.py` อย่างละเอียด เพื่อให้เข้าใจถึงวิธีการที่ไฟล์นี้ทำหน้าที่เป็น **รากฐานในการนิยามและสร้าง Event Objects** จากข้อมูลดิบที่ได้จาก PyTorch Profiler

## ภาพรวมของ `trace.py`

`trace.py` ไม่ได้ทำการวิเคราะห์ข้อมูลที่ซับซ้อน แต่ทำหน้าที่ที่สำคัญอย่างยิ่งคือ **การแปลงข้อมูลดิบ (raw trace data) ให้เป็น object ที่มีโครงสร้างและมีความหมาย (typed Event objects)** ซึ่งเป็นข้อมูลตั้งต้นที่ parser อื่นๆ (เช่น `event_parser.py`, `data.py`) จะนำไปใช้ประมวลผลต่อ

หน้าที่หลักของไฟล์นี้แบ่งได้เป็น 3 ส่วน:
1.  **นิยาม (Define):** กำหนดโครงสร้างข้อมูลพื้นฐาน เช่น ประเภทของ Event, ประเภทของ Device, และชื่อของ Communication operations
2.  **สร้าง (Construct):** สร้างคลาส (class) สำหรับ Event แต่ละประเภท เพื่อใช้เป็น "พิมพ์เขียว" ในการเก็บข้อมูล
3.  **แปลง (Transform):** สร้างฟังก์ชัน (factory functions) ที่รับข้อมูลดิบจาก JSON trace แล้วแปลงให้เป็น instance ของคลาส Event ที่เหมาะสม

---

## การทำงานของโค้ดแต่ละส่วน

### ส่วนที่ 1: การนิยามข้อมูลพื้นฐาน

```python
# pyre-unsafe
from enum import IntEnum
from typing import Dict, Optional

from .. import utils

__all__ = ['EventTypes', 'create_event']

logger = utils.get_logger()

NcclOpNameSet = ['nccl:broadcast', 'nccl:reduce', 'nccl:all_reduce', 'nccl:all_gather', 'nccl:reduce_scatter']
GlooOpNameSet = ['gloo:broadcast', 'gloo:reduce', 'gloo:all_reduce', 'gloo:all_gather', 'gloo:reduce_scatter']

class DeviceType(IntEnum):
    CPU = 0
    CUDA = 1
```

*   **`__all__ = [...]`**: เป็นมาตรฐานของ Python ที่บอกว่าเมื่อมีการ `import *` จากไฟล์นี้ จะ import แค่ `EventTypes` และ `create_event`
*   **`NcclOpNameSet`, `GlooOpNameSet`**:
    *   **หน้าที่**: กำหนด list ของชื่อ operation ที่เป็นการสื่อสารข้อมูล (communication) แบบ distributed โดยใช้ library NCCL และ Gloo
    *   **ผลลัพธ์**: `set` ของชื่อเหล่านี้จะถูกนำไปใช้ใน `event_parser.py` เพื่อตรวจสอบว่า event ใดเป็น communication event
*   **`DeviceType(IntEnum)`**:
    *   **หน้าที่**: สร้าง enum (ชุดของค่าคงที่) เพื่อใช้แทนประเภทของ device ทำให้โค้ดอ่านง่ายและผิดพลาดน้อยลง แทนที่จะใช้เลข 0 หรือ 1 ตรงๆ
    *   **ผลลัพธ์**: ได้ `DeviceType.CPU` และ `DeviceType.CUDA` สำหรับใช้ระบุประเภทของ device ใน `MemoryEvent`

### ส่วนที่ 2: การสร้างคลาสสำหรับ Event แต่ละประเภท

#### `class EventTypes:` และ `EventTypeMap`

```python
class EventTypes:
    TRACE = 'Trace'
    OPERATOR = 'Operator'
    # ... (และอื่นๆ) ...

EventTypeMap = {
    'trace': EventTypes.TRACE,
    'cpu_op': EventTypes.OPERATOR,
    # ... (และอื่นๆ) ...
}
```

*   **`EventTypes`**:
    *   **หน้าที่**: เป็นคลาสที่รวบรวมชื่อประเภทของ event ทั้งหมดที่ profiler นี้รู้จักไว้ในที่เดียว
    *   **ผลลัพธ์**: ทำให้สามารถอ้างอิงถึงประเภทของ event ด้วยชื่อที่สื่อความหมายได้ เช่น `EventTypes.OPERATOR` แทนที่จะใช้ string "Operator" ตรงๆ ลดโอกาสพิมพ์ผิด
*   **`EventTypeMap`**:
    *   **หน้าที่**: สร้าง dictionary สำหรับ "แปล" หรือ "map" ค่า `cat` (category) ที่มาจากไฟล์ JSON trace (ซึ่งเป็น string ตัวเล็ก) ไปเป็นประเภท `EventTypes` ที่กำหนดไว้
    *   **ผลลัพธ์**: เป็น map ที่ฟังก์ชัน `create_trace_event` จะใช้เพื่อหาว่า event ดิบควรจะถูกแปลงเป็น object ประเภทใด

#### `class BaseEvent:` และคลาสที่สืบทอด

```python
class BaseEvent:
    def __init__(self, type, data):
        self.type: str = type
        self.name: str = data.get('name')
        self.ts: int = data.get('ts')
        self.pid: int = data.get('pid')
        self.tid: int = data.get('tid')
        self.args: Dict = data.get('args', {})

class DurationEvent(BaseEvent):
    def __init__(self, type, data):
        super().__init__(type, data)
        self.category: str = data.get('cat', '')
        self.duration: int = data.get('dur')
        # ... (ดึง external_id, correlation_id) ...

class KernelEvent(DurationEvent):
    # ... (ดึงข้อมูลเฉพาะของ kernel เช่น occupancy, grid, block) ...

class OperatorEvent(DurationEvent):
    # ... (ดึงข้อมูลเฉพาะของ operator เช่น callstack, input_shape) ...

# ... (และคลาสอื่นๆ เช่น MemoryEvent, ModuleEvent) ...
```

*   **`BaseEvent`**:
    *   **หน้าที่**: เป็นคลาสแม่ (base class) สำหรับ event ทุกประเภท กำหนด attribute พื้นฐานที่ event ทุกตัวต้องมี
    *   **ผลลัพธ์**: `__init__` ของคลาสนี้จะดึงข้อมูลพื้นฐานจาก `data` (dictionary จาก JSON) เช่น `name`, `ts` (timestamp), `pid` (process ID), `tid` (thread ID), และ `args` (arguments เพิ่มเติม)
*   **`DurationEvent(BaseEvent)`**:
    *   **หน้าที่**: เป็นคลาสแม่สำหรับ event ที่มี "ระยะเวลา" (duration)
    *   **ผลลัพธ์**: สืบทอดคุณสมบัติจาก `BaseEvent` และเพิ่มการดึงข้อมูล `dur` (duration), `cat` (category), `external id` (สำหรับเชื่อมกับ runtime), และ `correlation` (สำหรับเชื่อมกับ device event)
*   **`KernelEvent(DurationEvent)`, `OperatorEvent(DurationEvent)`, etc.**:
    *   **หน้าที่**: เป็นคลาสเฉพาะสำหรับ event แต่ละประเภท
    *   **ผลลัพธ์**: สืบทอดคุณสมบัติจาก `DurationEvent` และเพิ่มการดึงข้อมูลเฉพาะทางจาก `args` ของ event นั้นๆ
        *   `KernelEvent` จะดึงข้อมูลเกี่ยวกับ GPU kernel เช่น `blocks per SM`, `occupancy`
        *   `OperatorEvent` จะดึงข้อมูลเกี่ยวกับ PyTorch operator เช่น `Call stack`, `Input Dims`
        *   `MemoryEvent` จะดึงข้อมูลเกี่ยวกับการใช้ memory เช่น `Device Id`, `Bytes`
        *   `ModuleEvent` จะดึงข้อมูลเกี่ยวกับ `nn.Module` เช่น `Python module id`

### ส่วนที่ 3: ฟังก์ชันสำหรับสร้าง Event Object (Factory Functions)

#### `def create_event(event, is_pytorch_lightning)`

```python
def create_event(event, is_pytorch_lightning) -> Optional[BaseEvent]:
    try:
        type = event.get('ph')
        if type == 'X':
            return create_trace_event(event, is_pytorch_lightning)
        elif type == 'i' and event.get('name') == '[memory]':
            return MemoryEvent(EventTypes.MEMORY, event)
        else:
            return None
    # ... (Exception handling) ...
```
*   **หน้าที่**: เป็นฟังก์ชัน "โรงงาน" (Factory) หลักที่รับ event ดิบ (`event`) เข้ามา แล้วตัดสินใจว่าจะสร้าง object ประเภทใด
*   **การทำงาน**:
    1.  ดูค่า `ph` (phase) ของ event จากไฟล์ trace:
        *   ถ้า `ph == 'X'`: หมายถึง event ที่มีระยะเวลา (Complete Event) จะส่งต่อไปให้ `create_trace_event` จัดการ
        *   ถ้า `ph == 'i'` และ `name == '[memory]'`: หมายถึง event ที่บอกข้อมูล memory (Instant Event) จะสร้าง `MemoryEvent`
        *   ถ้าเป็นประเภทอื่น: จะไม่สร้าง object ใดๆ (`return None`)
*   **ผลลัพธ์**: คืนค่าเป็น instance ของคลาส Event ที่เหมาะสม หรือ `None` ถ้าเป็น event ที่ไม่สนใจ

#### `def create_trace_event(event, is_pytorch_lightning)`

```python
def create_trace_event(event, is_pytorch_lightning) -> Optional[BaseEvent]:
    category = event.get('cat')
    event_type = EventTypeMap.get(category.lower())
    if event_type == EventTypes.OPERATOR:
        name = event.get('name')
        if name and name.startswith('ProfilerStep#'):
            return ProfilerStepEvent(event)
        # ... (จัดการกรณี PyTorch Lightning) ...
        return OperatorEvent(event_type, event)
    elif event_type == EventTypes.KERNEL:
        return KernelEvent(event_type, event)
    elif event_type == EventTypes.PYTHON_FUNCTION:
        args = event.get('args')
        if args and args.get('Python module id') is not None:
            return ModuleEvent(event)
        else:
            return PythonFunctionEvent(event_type, event)
    # ... (เงื่อนไขอื่นๆ) ...
    return None
```
*   **หน้าที่**: เป็นฟังก์ชันโรงงานสำหรับ event ที่มีระยะเวลา (`ph == 'X'`)
*   **การทำงาน**:
    1.  `category = event.get('cat')`: ดึงค่า category ของ event
    2.  `event_type = EventTypeMap.get(category.lower())`: ใช้ `EventTypeMap` เพื่อแปล `category` เป็น `EventTypes` ภายใน
    3.  ใช้ `if-elif-else` ขนาดใหญ่เพื่อตรวจสอบ `event_type` และข้อมูลอื่นๆ (เช่น `name`, `args`) เพื่อตัดสินใจว่าจะสร้าง object ของคลาสใดที่เฉพาะเจาะจงที่สุด
        *   ถ้า `event_type` คือ `OPERATOR` และชื่อขึ้นต้นด้วย `ProfilerStep#` -> สร้าง `ProfilerStepEvent`
        *   ถ้า `event_type` คือ `OPERATOR` ทั่วไป -> สร้าง `OperatorEvent`
        *   ถ้า `event_type` คือ `KERNEL` -> สร้าง `KernelEvent`
        *   ถ้า `event_type` คือ `PYTHON_FUNCTION` และมี `Python module id` ใน `args` -> สร้าง `ModuleEvent`
        *   ถ้าไม่ตรงกับเงื่อนไขเฉพาะใดๆ แต่มี `event_type` -> สร้าง `DurationEvent` (เป็น default)
*   **ผลลัพธ์**: คืนค่าเป็น instance ของคลาส Event ที่เฉพาะเจาะจงที่สุดสำหรับ event ดิบนั้นๆ

#### `def create_association_events(events)`

```python
def create_association_events(events) -> Dict[int, int]:
    forward_map = {}
    backward_map = {}

    result = {}
    for e in events:
        ph = e.get('ph')
        id = e['id']
        ts = e['ts']
        if ph == 's':
            forward_map[id] = ts
        elif ph == 'f':
            backward_map[id] = ts

    for id, ts in forward_map.items():
        backward_ts = backward_map.get(id)
        if backward_ts is not None:
            result[ts] = backward_ts

    return result
```
*   **หน้าที่**: **สำคัญที่สุดสำหรับ Forward/Backward Pass** - สร้าง dictionary ที่เชื่อมโยง forward event กับ backward event ที่สอดคล้องกัน
*   **การทำงาน**:
    1.  `forward_map`, `backward_map`: สร้าง dictionary ว่างสำหรับเก็บข้อมูล
    2.  วนลูปผ่าน `events` (ซึ่งถูกกรองมาแล้วให้มีเฉพาะ `cat='fwdbwd'`):
        *   `ph = e.get('ph')`: ดึง phase ของ event
        *   `id = e['id']`: ดึง **correlation ID** ที่ PyTorch ใส่มาเพื่อจับคู่ event
        *   ถ้า `ph == 's'` (start of async flow): เก็บ `id` และ `ts` (timestamp) ลงใน `forward_map`
        *   ถ้า `ph == 'f'` (finish of async flow): เก็บ `id` และ `ts` ลงใน `backward_map`
    3.  วนลูปผ่าน `forward_map` ที่สร้างเสร็จแล้ว:
        *   สำหรับแต่ละ `id` และ `ts` (forward timestamp), จะไปค้นหา `id` เดียวกันใน `backward_map` เพื่อหา `backward_ts`
        *   ถ้าเจอ, จะสร้าง entry ใน `result` dictionary โดยมี **key เป็น timestamp ของ forward event** และ **value เป็น timestamp ของ backward event**
*   **ผลลัพธ์ (`fwd_bwd_map`)**: คืนค่า dictionary (`result`) ที่เป็น "แผนที่" เชื่อมโยง forward pass กับ backward pass ซึ่ง `op_tree.py` จะนำไปใช้สร้าง Op Tree ที่สมบูรณ์

---
หวังว่าคำอธิบายนี้จะช่วยให้เข้าใจบทบาทและการทำงานของ `trace.py` ในฐานะรากฐานของการประมวลผลข้อมูลของ profiler ได้อย่างละเอียดและชัดเจนครับ
