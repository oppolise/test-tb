# เอกสารอธิบายไฟล์ run.py

ไฟล์ `run.py` ใน `torch_tb_profiler` เป็นส่วนสำคัญที่จัดการกับข้อมูลโปรไฟล์ที่ถูกประมวลผลและเตรียมไว้สำหรับการแสดงผลใน TensorBoard ประกอบด้วยคลาสหลักสองคลาสคือ `Run` และ `RunProfile` (รวมถึง `DistributedRunProfile`) ซึ่งทำหน้าที่จัดโครงสร้าง, จัดเก็บ, และให้บริการข้อมูลโปรไฟล์ในรูปแบบต่างๆ

## ภาพรวม

-   **`Run`**: คลาสนี้แทน "run" หนึ่งๆ ของการโปรไฟล์ ซึ่งอาจประกอบด้วยข้อมูลจากหลาย "worker" (เช่น ในกรณีของ distributed training) `Run` จะเก็บ `RunProfile` ของแต่ละ worker/span
-   **`RunProfile`**: คลาสนี้เก็บข้อมูลโปรไฟล์ที่ "ปรุงสุก" (cooked) แล้วสำหรับ worker หนึ่งๆ ใน span (ช่วงเวลา) ที่กำหนด ข้อมูลนี้รวมถึงภาพรวม (overview), สถิติของ operation และ kernel, ข้อมูล trace, ข้อมูลหน่วยความจำ, และข้อมูล module view
-   **`DistributedRunProfile`**: คลาสพิเศษสำหรับมุมมองแบบ distributed ซึ่งรวมข้อมูลจากทุก worker ใน span ที่กำหนด

## การนำเข้าโมดูล (Import Modules)

-   `collections.defaultdict`: สำหรับสร้าง dictionary ที่มีค่าเริ่มต้นสำหรับ key ใหม่
-   `typing`: สำหรับ type hinting เพื่อเพิ่มความชัดเจนของโค้ด
-   `. import consts, utils`: นำเข้าค่าคงที่และฟังก์ชัน utilities ภายใน
-   `.profiler.diffrun import compare_op_tree, diff_summary`: ฟังก์ชันสำหรับการเปรียบเทียบ (diff) ระหว่าง run สอง run
-   `.profiler.memory_parser import MemoryMetrics, MemoryRecord, MemorySnapshot`: คลาสและ enum สำหรับการจัดการข้อมูลหน่วยความจำ
-   `.profiler.module_op import Stats`: namedtuple สำหรับเก็บสถิติของ module
-   `.profiler.node import OperatorNode`: คลาสสำหรับแทน operator ในรูปแบบ tree structure
-   `.utils import Canonicalizer, DisplayRounder, lttb_sample`: utility classes สำหรับการแปลงหน่วย (canonicalize), การปัดเศษตัวเลข, และการสุ่มตัวอย่างข้อมูล (Largest-Triangle-Three-Buckets)

## คลาส `Run`

คลาส `Run` ทำหน้าที่เป็น container สำหรับข้อมูลโปรไฟล์จากหนึ่ง "run" ซึ่งอาจมีหลาย worker และหลาย span

```python
class Run:
    def __init__(self, name, run_dir):
        self.name = name  # ชื่อของ run
        self.run_dir = run_dir  # ไดเรกทอรีของ run นี้
        self.profiles: Dict[Tuple[str, str], RunProfile] = {} # เก็บ RunProfile โดยมี key เป็น (worker_name, span_name)
```

### Properties

-   `workers(self)`:
    -   คืนค่าลิสต์ของชื่อ worker ทั้งหมดที่มีใน run นี้ (ไม่ซ้ำและเรียงตามตัวอักษร)
-   `views(self) -> List[consts.View]`:
    -   คืนค่าลิสต์ของ `consts.View` (เช่น Overview, Operator View) ที่มีอยู่ในโปรไฟล์ใดๆ ของ run นี้ (ไม่ซ้ำและเรียงตาม ID ของ view)

### เมธอด

-   `get_workers(self, view)`:
    -   คืนค่าลิสต์ของชื่อ worker ที่มี view ที่ระบุ (เรียงตามตัวอักษร)
-   `get_spans(self, worker=None)`:
    -   คืนค่าลิสต์ของชื่อ span ที่มี ถ้า `worker` ถูกระบุ จะคืน span ของ worker นั้นๆ มิฉะนั้นจะคืน span ทั้งหมดที่มีใน run (ไม่ซ้ำและเรียงตามตัวอักษร)
    -   ถ้ามีแค่ span เดียวและเป็น `None` (ค่าเริ่มต้น) จะคืน `None`
-   `add_profile(self, profile: Union['DistributedRunProfile', 'RunProfile'])`:
    -   เพิ่ม `RunProfile` หรือ `DistributedRunProfile` เข้าไปใน `self.profiles`
    -   ใช้ชื่อ worker และชื่อ span (ถ้า span เป็น `None` จะใช้ 'default') เป็น key
-   `get_profile(self, worker, span) -> Union['DistributedRunProfile', 'RunProfile']`:
    -   ดึง `RunProfile` หรือ `DistributedRunProfile` สำหรับ worker และ span ที่ระบุ
    -   คืนค่า `None` ถ้าไม่พบ
    -   `worker` เป็นพารามิเตอร์บังคับ
-   `get_profiles(self, *, worker=None, span=None) -> Optional[Union[List['RunProfile'], List['DistributedRunProfile']]]`:
    -   ดึงรายการโปรไฟล์:
        -   ถ้า `worker` และ `span` ระบุ: คืนโปรไฟล์เดียว (หรือ `None`)
        -   ถ้าเฉพาะ `worker` ระบุ: คืนลิสต์ของโปรไฟล์ทั้งหมดสำหรับ worker นั้น
        -   ถ้าเฉพาะ `span` ระบุ: คืนลิสต์ของโปรไฟล์ทั้งหมดสำหรับ span นั้น
        -   ถ้าไม่ระบุทั้งคู่: คืนลิสต์ของโปรไฟล์ทั้งหมดใน run นี้

## คลาส `RunProfile`

คลาส `RunProfile` เก็บข้อมูลโปรไฟล์ที่ประมวลผลแล้วสำหรับ worker หนึ่งๆ ใน span (ช่วงเวลา) ที่กำหนด ข้อมูลนี้พร้อมสำหรับการแสดงผลใน TensorBoard frontend

```python
class RunProfile:
    def __init__(self, worker, span):
        self.worker = worker  # ชื่อ worker
        self.span = span      # ชื่อ span (อาจเป็น None)
        self.views: List[consts.View] = []  # รายการ view ที่มีในโปรไฟล์นี้
        self.is_pytorch_lightning = False   # Flag ว่าเป็น PyTorch Lightning run หรือไม่
        self.has_runtime = False            # Flag ว่ามีข้อมูล runtime (CPU ops) หรือไม่
        self.has_kernel = False             # Flag ว่ามีข้อมูล kernel (GPU ops) หรือไม่
        self.has_communication = False      # Flag ว่ามีข้อมูล communication ops (สำหรับ distributed) หรือไม่
        self.has_memcpy_or_memset = False # Flag ว่ามีข้อมูล memcpy/memset หรือไม่
        self.profiler_start_ts = float('inf') # Timestamp เริ่มต้นของ profiler
        self.overview = None                # ข้อมูลสำหรับ Overview tab
        # ข้อมูลสำหรับ Operator tab (pie chart, table, stack trace)
        self.operation_pie_by_name = None
        self.operation_table_by_name = None
        self.operation_stack_by_name: Dict = None
        self.operation_pie_by_name_input = None # Групування за назвою операції та формою вхідних даних
        self.operation_table_by_name_input = None
        self.operation_stack_by_name_input: Dict = None
        # ข้อมูลสำหรับ Kernel tab (pie chart, table)
        self.kernel_op_table = None         # Kernel table grouped by operator
        self.kernel_pie = None
        self.kernel_table = None
        self.tc_pie = None                  # TensorCore usage pie chart
        self.trace_file_path: str = None   # Path ไปยังไฟล์ trace ดั้งเดิม

        self.gpu_metrics = None             # ข้อมูล GPU metrics (ถ้ามี)

        self.gpu_summary = None             # สรุปข้อมูล GPU (สำหรับ Overview)
        self.gpu_tooltip = None             # Tooltip สำหรับ GPU summary

        # สำหรับ Memory View
        self.memory_snapshot: Optional[MemorySnapshot] = None # Snapshot ของข้อมูล memory
        self.tid2tree: Dict[int, OperatorNode] = None # Mapping จาก thread ID ไปยัง OperatorNode tree (สำหรับ CPU ops)
        self.pl_tid2tree: Dict[int, OperatorNode] = None # เหมือน tid2tree แต่สำหรับ PyTorch Lightning

        # สำหรับ Module View
        self.module_stats: Optional[List(Stats)] = None # สถิติของ module
        self.pl_module_stats: Optional[List(Stats)] = None # สถิติ module สำหรับ PyTorch Lightning
```

### เมธอด `append_gpu_metrics(self, raw_data: bytes)`

-   **วัตถุประสงค์**: เพิ่มข้อมูล GPU metrics (ที่เก็บใน `self.gpu_metrics`) เข้าไปในข้อมูล trace ดั้งเดิม (`raw_data`)
-   **การทำงาน**:
    -   แปลง `self.gpu_metrics` (ซึ่งควรเป็น list ของ JSON strings) ให้เป็น JSON array string
    -   หาตำแหน่งของ `]` สุดท้ายใน `raw_data` (ซึ่งเป็น JSON array ของ trace events)
    -   ต่อ string ของ GPU metrics เข้าไปก่อน `]` สุดท้าย และปิดท้ายด้วย `}]` เพื่อให้เป็น JSON ที่ถูกต้อง
    -   บีบอัดข้อมูลที่ได้ด้วย `gzip`
-   **การใช้งาน**: ถูกเรียกโดย `TorchProfilerPlugin.trace_route` เมื่อต้องการส่งข้อมูล trace ที่มี GPU metrics ไปยัง frontend

### เมธอด `_filtered_by_ts(events: Iterable[MemoryRecord], start_ts, end_ts)` (Static Method)

-   **วัตถุประสงค์**: กรองรายการ `MemoryRecord` ให้อยู่ในช่วง timestamp ที่กำหนด (`start_ts` ถึง `end_ts`)
-   **การทำงาน**: คืนค่าลิสต์ใหม่ของ `MemoryRecord` ที่ผ่านการกรอง

### เมธอด `get_memory_stats(self, start_ts=None, end_ts=None, memory_metric='K')`

-   **วัตถุประสงค์**: สร้างข้อมูลสถิติหน่วยความจำสำหรับ Memory View (ตารางสรุป)
-   **การทำงาน**:
    1.  สร้าง instance ของ `Canonicalizer` (สำหรับแปลงหน่วย) และ `DisplayRounder` (สำหรับปัดเศษ)
    2.  เรียก `self.memory_snapshot.get_memory_statistics()` เพื่อคำนวณสถิติหน่วยความจำโดยอิงจาก `self.tid2tree` และช่วงเวลาที่กำหนด
    3.  จัดรูปแบบข้อมูลที่ได้ให้อยู่ในโครงสร้างที่ frontend ต้องการ ซึ่งประกอบด้วย:
        -   `metadata`: ชื่อ title, default device, ชื่อคอลัมน์สำหรับ search/sort
        -   `columns`: รายละเอียดของแต่ละคอลัมน์ (ชื่อ, ประเภท, tooltip)
        -   `rows`: ข้อมูลสถิติของแต่ละ operator สำหรับแต่ละ device (เช่น 'CPU', 'GPU0')
            -   Operator Name
            -   Calls (จำนวนครั้งที่เรียก)
            -   Size Increase (ขนาดหน่วยความจำที่เพิ่มขึ้นรวม children)
            -   Self Size Increase (ขนาดหน่วยความจำที่เพิ่มขึ้นเฉพาะตัว operator เอง)
            -   Allocation Count (จำนวนการ allocate รวม children)
            -   Self Allocation Count (จำนวนการ allocate เฉพาะตัว operator เอง)
            -   Allocation Size (ขนาดหน่วยความจำที่ allocate รวม children)
            -   Self Allocation Size (ขนาดหน่วยความจำที่ allocate เฉพาะตัว operator เอง)
    4.  กำหนด default device โดยดูจาก device ที่มี (ถ้ามี GPU จะเลือก GPU แรก)
-   **คืนค่า**: Dictionary ที่มีข้อมูลพร้อมสำหรับแสดงผลใน Memory View (ตาราง)

### เมธอด `get_memory_curve(self, time_metric: str = 'ms', memory_metric: str = 'K', patch_for_step_plot=True)`

-   **วัตถุประสงค์**: สร้างข้อมูลสำหรับแสดงกราฟการใช้หน่วยความจำ (Memory Curve)
-   **การทำงาน**:
    1.  `get_curves_and_peaks`:
        -   ประมวลผล `self.memory_snapshot.memory_records`
        -   สำหรับแต่ละ device ('CPU', 'GPU0', ...):
            -   สร้างลิสต์ของ data points `[timestamp, total_allocated, total_reserved]`
            -   หาค่า peak memory usage
        -   แปลงหน่วยเวลาและหน่วยความจำโดยใช้ `Canonicalizer`
    2.  `patch_curves_for_step_plot` (optional):
        -   ปรับแก้ข้อมูล curve เพื่อให้ line plot แสดงผลเหมือน step plot (โดยการเพิ่มจุดข้อมูลซ้ำที่ตำแหน่ง x ใหม่ แต่ y เดิม)
    3.  จัดรูปแบบข้อมูล peaks และ totals (ถ้ามีข้อมูล memory ทั้งหมดของ GPU)
    4.  กำหนด default device
    5.  ใช้ `lttb_sample` เพื่อลดจำนวนจุดข้อมูลใน curve (downsampling) โดยยังคงรูปร่างของกราฟไว้
    6.  จัดโครงสร้างข้อมูลสุดท้ายสำหรับ frontend:
        -   `metadata`: default device, รายชื่อ devices, ข้อความ peak memory, first timestamp, หน่วยที่ใช้
        -   `columns`: คำอธิบายคอลัมน์ (Time, Allocated, Reserved)
        -   `rows`: ข้อมูล curve ที่ผ่านการ downsample แล้ว (ในรูปแบบ dictionary `{'CPU': [...], 'GPU0': [...]}`)
-   **คืนค่า**: Dictionary ที่มีข้อมูลพร้อมสำหรับแสดงผลกราฟ Memory Curve

### เมธอด `get_memory_events(self, start_ts=None, end_ts=None, time_metric: str = 'ms', memory_metric: str = 'K')`

-   **วัตถุประสงค์**: สร้างรายการของ memory allocation/deallocation events สำหรับ Memory Events View
-   **การทำงาน**:
    1.  `get_op_name_or_ctx`: Helper function เพื่อดึงชื่อ operator (ถ้าเป็น `aten::empty` จะพยายามรวมชื่อ parent operator เข้าไปด้วย)
    2.  กรอง `self.memory_snapshot.memory_records` ตาม `start_ts` และ `end_ts`
    3.  วนลูปผ่าน memory records:
        -   จับคู่ allocation (`is_allocation=True`) กับ deallocation (`is_allocation=False`) event โดยใช้ `addr` (memory address)
        -   สำหรับคู่ที่สมบูรณ์ (มีทั้ง alloc และ free): สร้าง event ที่มีข้อมูล [Operator, Size (ลบ), Alloc Time, Release Time, Duration]
        -   สำหรับ allocation ที่ไม่มีคู่ (ยังไม่ถูก free): สร้าง event ที่มีข้อมูล [Operator, Size (บวก), Alloc Time, None, None]
        -   สำหรับ deallocation ที่ไม่มีคู่ (ไม่พบ allocation ก่อนหน้า): สร้าง event ที่มีข้อมูล [Operator, Size (ลบ), None, Release Time, None] (และ log warning)
    4.  จัดกลุ่ม events ตาม device
    5.  กำหนด default device
    6.  จัดโครงสร้างข้อมูลสุดท้ายสำหรับ frontend:
        -   `metadata`: title, default device
        -   `columns`: คำอธิบายคอลัมน์ (Operator, Size, Allocation Time, Release Time, Duration)
        -   `rows`: ข้อมูล events ที่จัดกลุ่มตาม device
-   **คืนค่า**: Dictionary ที่มีข้อมูลพร้อมสำหรับแสดงผลใน Memory Events View

### เมธอด `get_module_view(self)`

-   **วัตถุประสงค์**: สร้างข้อมูลสำหรับ Module View (แสดงสถิติตาม `nn.Module`)
-   **การทำงาน**:
    1.  เลือก `module_stats` ที่จะใช้ (ถ้าเป็น PyTorch Lightning run และมี `pl_module_stats` จะใช้ตัวนั้น)
    2.  ถ้าไม่มี `module_stats` จะคืนค่า `None`
    3.  จัดโครงสร้างข้อมูลสำหรับ frontend:
        -   `columns`: คำอธิบายคอลัมน์ (Module Name, Occurrences, Operators, Host Total/Self Time, Device Total/Self Time)
        -   `data`: ข้อมูลสถิติของแต่ละ module ในรูปแบบ tree structure (ใช้ `process_modules_stats` ซึ่งเป็น recursive helper function)
-   **คืนค่า**: Dictionary ที่มีข้อมูลพร้อมสำหรับแสดงผลใน Module View, หรือ `None`

### เมธอด `get_operator_tree(self)`

-   **วัตถุประสงค์**: สร้างข้อมูล Operator Tree View (แสดง call hierarchy ของ operators)
-   **การทำงาน**:
    1.  เลือก root node ของ operator tree (จาก `self.tid2tree` หรือ `self.pl_tid2tree` ถ้าเป็น PyTorch Lightning)
    2.  ใช้ `traverse_node` (recursive helper function) เพื่อแปลง `OperatorNode` tree ให้อยู่ในรูปแบบ JSON ที่ frontend ต้องการ (แต่ละ node มี name, start_time, end_time, type, tid, children)
-   **คืนค่า**: Dictionary ที่แทน root ของ operator tree

### เมธอด `compare_run(self, exp: 'RunProfile')`

-   **วัตถุประสงค์**: เปรียบเทียบ `RunProfile` ปัจจุบัน (base) กับ `RunProfile` อื่น (exp)
-   **การทำงาน**:
    1.  ดึง root node ของ operator tree จาก `self.tid2tree` (สำหรับ base) และ `exp.tid2tree` (สำหรับ experiment)
    2.  เรียก `compare_op_tree(base_root, exp_root)` (จาก `diffrun.py`) เพื่อสร้าง diff tree
    3.  เรียก `diff_summary(diff_root)` (จาก `diffrun.py`) เพื่อสร้างสรุปสถิติของ diff
-   **คืนค่า**: `DiffStats` object ที่เก็บผลการเปรียบเทียบ

## คลาส `DistributedRunProfile`

คลาสนี้ใช้สำหรับเก็บข้อมูลโปรไฟล์ในมุมมองแบบ distributed ซึ่งรวมข้อมูลจาก "All" workers สำหรับ span ที่กำหนด

```python
class DistributedRunProfile:
    def __init__(self, span: str):
        self.worker = 'All'  # Worker name ถูกตั้งค่าเป็น 'All' เสมอ
        self.span = span     # ชื่อ span
        self.views = []      # รายการ view ที่มี (จะถูกเติมโดย RunLoader)
        # ข้อมูลเฉพาะสำหรับ distributed view
        self.gpu_info = None           # ข้อมูล GPU
        self.steps_to_overlap = None   # ข้อมูล computation/communication overlap
        self.steps_to_wait = None      # ข้อมูล communication wait time
        self.comm_ops = None           # ข้อมูล communication operations
```

-   คลาสนี้มีโครงสร้างที่ง่ายกว่า `RunProfile` เนื่องจากเน้นที่ข้อมูลสรุประดับ distributed
-   Attribute ต่างๆ เช่น `gpu_info`, `steps_to_overlap` จะถูกเติมค่าโดย `RunLoader` เมื่อประมวลผลข้อมูล distributed

## สรุป

ไฟล์ `run.py` ทำหน้าที่เป็น data model หลักสำหรับข้อมูลโปรไฟล์ที่ประมวลผลแล้ว คลาส `Run` จัดการกลุ่มของโปรไฟล์ (จาก worker/span ต่างๆ) ในขณะที่ `RunProfile` จัดการรายละเอียดของโปรไฟล์เดียว รวมถึงการแปลงข้อมูลให้อยู่ในรูปแบบที่ frontend ของ TensorBoard สามารถนำไปแสดงผลได้โดยตรงสำหรับ view ต่างๆ เช่น Overview, Operator, Kernel, Memory, Module, และ Trace.
