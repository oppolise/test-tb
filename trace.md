# คำอธิบายโค้ด `profiler/trace.py` เชิงลึก

เอกสารนี้อธิบายการทำงานของไฟล์ `trace.py` จาก PyTorch Profiler Plugin อย่างละเอียด เพื่อให้เกิดความเข้าใจในวัตถุประสงค์, โครงสร้างข้อมูล, และตรรกะการทำงานของแต่ละส่วน สำหรับนำไปศึกษาและปรับใช้

## ภาพรวม

ไฟล์ `trace.py` ทำหน้าที่เป็น **"Data Definition and Factory Layer"** สำหรับ profiler events. วัตถุประสงค์หลักของมันคือการแปลง event ดิบจากไฟล์ JSON (ซึ่งอยู่ในรูปแบบ dictionary) ให้กลายเป็น Python object ที่มีประเภทและ attributes ที่ชัดเจน (typed objects). การทำเช่นนี้ช่วยให้ส่วนอื่นๆ ของโค้ด (เช่น `EventParser`, `MemoryParser`) สามารถนำข้อมูลไปใช้งานได้อย่างปลอดภัย, สะดวก, และอ่านง่ายขึ้น.

นอกจากนี้ `trace.py` ยังมีหน้าที่สำคัญในการประมวลผล "flow events" เพื่อสร้างข้อมูลความสัมพันธ์ (association) ระหว่าง forward และ backward pass.

ไฟล์นี้แบ่งออกเป็น 4 ส่วนหลัก:
1.  **ค่าคงที่และ Enumerations**: กำหนดค่าคงที่ที่ใช้ร่วมกันทั่วทั้ง profiler.
2.  **คลาสสำหรับ Event (Event Classes)**: กำหนดโครงสร้างข้อมูลสำหรับ event แต่ละประเภท.
3.  **ฟังก์ชันสร้าง Event Object (Factory Functions)**: ฟังก์ชันสำหรับแปลง dictionary ดิบเป็น typed object.
4.  **ฟังก์ชันสร้างความสัมพันธ์ Forward/Backward**: ฟังก์ชันสำหรับสร้าง mapping ระหว่าง forward และ backward ops.

---

## ส่วนที่ 1: ค่าคงที่และ Enumerations

*   **วัตถุประสงค์**: กำหนดค่าคงที่ที่ใช้ร่วมกันเพื่อความสอดคล้องและลดความผิดพลาดจากการ hardcode string.
*   **คำอธิบายโค้ด**:

    ```python
    # List ของชื่อ operator ที่จัดว่าเป็น communication operations ของ NCCL
    NcclOpNameSet = ['nccl:broadcast', 'nccl:reduce', 'nccl:all_reduce', 'nccl:all_gather', 'nccl:reduce_scatter']
    # List ของชื่อ operator ที่จัดว่าเป็น communication operations ของ Gloo
    GlooOpNameSet = ['gloo:broadcast', 'gloo:reduce', 'gloo:all_reduce', 'gloo:all_gather', 'gloo:reduce_scatter']

    # Enum สำหรับระบุประเภทของ device (CPU = 0, CUDA = 1)
    class DeviceType(IntEnum):
        CPU = 0
        CUDA = 1

    # Class ที่รวบรวมชื่อ "ประเภท event" ที่เป็นมาตรฐานที่ใช้ภายใน profiler นี้
    # เพื่อให้โค้ดส่วนอื่นอ้างอิงถึงชื่อมาตรฐานเหล่านี้แทนการใช้ string โดยตรง
    class EventTypes:
        TRACE = 'Trace'
        OPERATOR = 'Operator'
        PROFILER_STEP = 'ProfilerStep'
        RUNTIME = 'Runtime'
        KERNEL = 'Kernel'
        # ... (และอื่นๆ)

    # Dictionary สำหรับ map ชื่อ category (สตริง) ที่มาจากไฟล์ trace ดิบ
    # ไปยัง EventTypes ที่เป็นมาตรฐานภายใน
    # เช่น 'cpu_op' และ 'operator' ในไฟล์ trace จะถูก map ไปเป็น EventTypes.OPERATOR เหมือนกัน
    EventTypeMap = {
        'trace': EventTypes.TRACE,
        'cpu_op': EventTypes.OPERATOR,
        'operator': EventTypes.OPERATOR,
        'runtime': EventTypes.RUNTIME,
        'kernel': EventTypes.KERNEL,
        # ... (และอื่นๆ)
    }
    ```
*   **ผลลัพธ์ของส่วนนี้**: สร้างค่าคงที่และ mapping ที่จะถูกนำไปใช้ในคลาสและฟังก์ชันต่อไป ทำให้โค้ดมีความน่าเชื่อถือและบำรุงรักษาง่ายขึ้น.

---

## ส่วนที่ 2: คลาสสำหรับ Event (Event Classes)

*   **วัตถุประสงค์**: สร้างพิมพ์เขียว (blueprint) สำหรับ event แต่ละประเภท โดยใช้หลักการ Object-Oriented Programming (Inheritance) เพื่อลดความซ้ำซ้อนและจัดระเบียบข้อมูล.
*   **คำอธิบายโค้ด**:

    ```python
    # คลาสแม่สำหรับ event ทั้งหมด
    class BaseEvent:
        def __init__(self, type, data):
            # type: ประเภทของ event จาก EventTypes (เช่น 'Operator', 'Kernel')
            self.type: str = type
            # data.get('...'): ดึงข้อมูลจาก dictionary ดิบของ event นั้นๆ
            self.name: str = data.get('name')
            self.ts: int = data.get('ts') # Timestamp
            self.pid: int = data.get('pid') # Process ID
            self.tid: int = data.get('tid') # Thread ID
            # args: dictionary ที่เก็บข้อมูลเพิ่มเติมของ event
            self.args: Dict = data.get('args', {})

    # คลาสสำหรับ event ที่มีระยะเวลา (สืบทอดคุณสมบัติจาก BaseEvent)
    class DurationEvent(BaseEvent):
        def __init__(self, type, data):
            super().__init__(type, data)
            # cat: category ดิบจากไฟล์ trace (เช่น 'cpu_op', 'kernel')
            self.category: str = data.get('cat', '')
            # dur: ระยะเวลาของ event
            self.duration: int = data.get('dur')
            # ดึง external id (อาจมี key เป็น 'external id' หรือ 'External id')
            # ใช้สำหรับเชื่อมโยง CPU op กับ GPU kernel ที่มัน launch
            extern_id: Optional[int] = self.args.get('external id')
            if extern_id is None:
                extern_id = self.args.get('External id')
            self.external_id = extern_id
            # ดึง correlation id (ใช้สำหรับเชื่อมโยง CUDA runtime calls กับ GPU kernels)
            self.correlation_id: Optional[int] = self.args.get('correlation')

    # คลาสสำหรับ Kernel event (สืบทอดจาก DurationEvent)
    class KernelEvent(DurationEvent):
        def __init__(self, type, data):
            super().__init__(type, data)
            # ดึงข้อมูลเฉพาะของ kernel จาก self.args
            self.occupancy = self.args.get('est. achieved occupancy %')
            self.blocks_per_sm = self.args.get('blocks per SM')
            self.grid = self.args.get('grid')
            self.block = self.args.get('block')
            self.regs_per_thread = self.args.get('registers per thread')
            self.shared_memory = self.args.get('shared memory')
            self.device_id = self.args.get('device')

    # คลาสสำหรับ Operator event (สืบทอดจาก DurationEvent)
    class OperatorEvent(DurationEvent):
        def __init__(self, type, data):
            super().__init__(type, data)
            # ดึงข้อมูลเฉพาะของ operator จาก self.args
            self.callstack = self.args.get('Call stack')
            self.input_type = self.args.get('Input type')
            # จัดการกับ key ที่อาจมีชื่อต่างกัน ('Input Dims' หรือ 'Input dims')
            shape = self.args.get('Input Dims')
            if shape is None:
                shape = self.args.get('Input dims', [])
            self.input_shape = shape

    # คลาสสำหรับ ProfilerStep event (สืบทอดจาก OperatorEvent)
    class ProfilerStepEvent(OperatorEvent):
        def __init__(self, data):
            super().__init__(EventTypes.PROFILER_STEP, data)
            # แปลงชื่อ "ProfilerStep#5" เป็น step number (int) คือ 5
            self.step: int = int(self.name.split('#')[1])

    # คลาสสำหรับ Memory event (สืบทอดจาก BaseEvent)
    class MemoryEvent(BaseEvent):
        def __init__(self, type, data):
            super().__init__(type, data)
            self.scope: str = data.get('s', '')
            self.device_id: int = self.args.get('Device Id')
            # ... (ดึงข้อมูลเฉพาะของ memory เช่น Bytes, Addr)
            # มีการแปลง Device Type ที่เป็น int ให้เป็น DeviceType Enum
        # ... (มี @property สำหรับเข้าถึงค่าใน self.args ได้ง่ายขึ้น)
    ```
    *   ... และคลาสอื่นๆ (`PythonFunctionEvent`, `ModuleEvent`, `PLProfileEvent`, `PLModuleEvent`) ที่มีลักษณะคล้ายกัน คือสืบทอดจากคลาสแม่และเพิ่ม attribute เฉพาะทางของตัวเอง.
*   **ผลลัพธ์ของส่วนนี้**: ได้โครงสร้างคลาสที่ชัดเจนสำหรับ event แต่ละประเภท พร้อมที่จะถูกสร้างเป็น object ในขั้นตอนถัดไป.

---

## ส่วนที่ 3: ฟังก์ชันสร้าง Event Object (Factory Functions)

*   **วัตถุประสงค์**: เป็นฟังก์ชันที่รับ dictionary ดิบจากไฟล์ trace และ "ผลิต" object ของคลาสที่เหมาะสมออกมา.
*   **คำอธิบายโค้ด**:

    ```python
    # Factory function หลัก ที่จะถูกเรียกจากภายนอก (โดย RunProfileData)
    def create_event(event, is_pytorch_lightning) -> Optional[BaseEvent]:
        try:
            # ph (phase) บอกลักษณะของ event ใน trace viewer format
            type = event.get('ph')
            # 'X' คือ complete event (มี start time และ duration)
            if type == 'X':
                # ส่งต่อให้ create_trace_event จัดการ
                return create_trace_event(event, is_pytorch_lightning)
            # 'i' คือ instant event (เกิดขึ้น ณ เวลาเดียว)
            elif type == 'i' and event.get('name') == '[memory]':
                return MemoryEvent(EventTypes.MEMORY, event)
            else:
                # ไม่รองรับ phase ประเภทอื่นในปัจจุบัน
                return None
        except Exception as ex:
            # จัดการ error หาก parse event ไม่สำเร็จ
            logger.warning('Failed to parse profile event. ...')
            raise

    # Factory function สำหรับ complete event ('X')
    def create_trace_event(event, is_pytorch_lightning) -> Optional[BaseEvent]:
        category = event.get('cat')
        # ใช้ EventTypeMap เพื่อแปลง category ดิบ (เช่น 'cpu_op') เป็น EventTypes มาตรฐาน (เช่น EventTypes.OPERATOR)
        event_type = EventTypeMap.get(category.lower())

        # ใช้ if-elif-else เพื่อสร้าง object ของคลาสที่ถูกต้องตาม event_type
        if event_type == EventTypes.USER_ANNOTATION:
            # จัดการกรณีพิเศษ: ถ้าเป็น user annotation แต่ชื่อเป็น "ProfilerStep#..."
            # ให้สร้างเป็น ProfilerStepEvent แทน
            name = event.get('name')
            if name and name.startswith('ProfilerStep#'):
                return ProfilerStepEvent(event)
            return OperatorEvent(event_type, event)
        elif event_type == EventTypes.OPERATOR:
            # ตรรกะคล้ายกันสำหรับ Operator, และมีการจัดการ event ของ PyTorch Lightning เพิ่มเติม
            name = event.get('name')
            if name and name.startswith('ProfilerStep#'):
                return ProfilerStepEvent(event)
            if is_pytorch_lightning:
                # ... (จัดการ PLProfileEvent, PLModuleEvent)
            return OperatorEvent(event_type, event)
        elif event_type == EventTypes.KERNEL:
            return KernelEvent(event_type, event)
        elif event_type == EventTypes.PYTHON_FUNCTION:
            # จัดการกรณีพิเศษ: ถ้าเป็น python_function แต่มี module id
            # ให้สร้างเป็น ModuleEvent แทน
            args = event.get('args')
            if args and args.get('Python module id') is not None:
                return ModuleEvent(event)
            else:
                return PythonFunctionEvent(event_type, event)
        elif event_type is not None:
            # กรณีทั่วไปสำหรับ event ที่มี duration อื่นๆ ที่ไม่มีคลาสเฉพาะ
            return DurationEvent(event_type, event)
        return None
    ```
*   **ผลลัพธ์ของส่วนนี้**: ฟังก์ชัน `create_event` ที่พร้อมจะถูกเรียกใช้โดย `RunProfileData` เพื่อแปลง list ของ dictionaries ให้เป็น list ของ typed objects.

---

## ส่วนที่ 4: ฟังก์ชันสร้างความสัมพันธ์ Forward/Backward

*   **วัตถุประสงค์**: ประมวลผล "flow" events (`cat: 'fwdbwd'`) เพื่อสร้าง mapping ที่เชื่อมโยง forward pass operations กับ backward pass operations ที่คู่กัน.
*   **คำอธิบายโค้ด**:

    ```python
    def create_association_events(events) -> Dict[int, int]:
        # forward_map เก็บ id -> timestamp ของ 'start' flow event
        forward_map = {}
        # backward_map เก็บ id -> timestamp ของ 'finish' flow event
        backward_map = {}
        result = {}

        # วนลูปผ่าน event ที่ถูกกรองมาแล้ว (เฉพาะ cat: 'fwdbwd')
        for e in events:
            ph = e.get('ph')
            id = e['id'] # Correlation ID ที่ Kineto ใส่มาให้เพื่อเชื่อมโยงกัน
            ts = e['ts'] # Timestamp
            if ph == 's': # 's' for start of a flow
                forward_map[id] = ts
            elif ph == 'f': # 'f' for finish of a flow
                backward_map[id] = ts

        # หลังจากรวบรวม map ทั้งสองแล้ว, ทำการจับคู่
        for id, ts in forward_map.items():
            # หา backward event ที่มี correlation id เดียวกัน
            backward_ts = backward_map.get(id)
            if backward_ts is not None:
                # ถ้าเจอ, สร้าง entry ใน result dictionary
                # โดย map timestamp ของ forward event ไปยัง timestamp ของ backward event
                result[ts] = backward_ts

        return result
    ```
*   **ผลลัพธ์ของส่วนนี้**: `result` dictionary ที่เป็นข้อมูลความสัมพันธ์ forward-backward ซึ่งจะถูกส่งต่อไปยัง `OpTreeBuilder` เพื่อใช้ในการสร้าง operator tree. นี่คือ "ข้อมูล" forward/backward ที่ถูกดึงออกมา.
