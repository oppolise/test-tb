# เอกสารอธิบายไฟล์ utils.py

ไฟล์ `utils.py` ใน `torch_tb_profiler` รวบรวมฟังก์ชันและคลาส tiện ích (utility) ต่างๆ ที่ใช้สนับสนุนการทำงานของส่วนอื่นๆ ในโปรไฟล์เลอร์ ฟังก์ชันเหล่านี้ช่วยในการจัดการ logging, การตรวจสอบประเภทไฟล์, การแปลงหน่วย, การจัดรูปแบบข้อมูลสำหรับการแสดงผล, การจับเวลา, และการสุ่มตัวอย่างข้อมูล

## การจัดการ Logger

```python
import logging
# ... (ส่วนอื่นๆ ของ imports)
from . import consts

def get_logging_level():
    log_level = os.environ.get('TORCH_PROFILER_LOG_LEVEL', 'INFO').upper()
    if log_level not in logging._levelToName.values():
        log_level = logging.getLevelName(logging.INFO)
    return log_level

logger = None

def get_logger():
    global logger
    if logger is None:
        logger = logging.getLogger(consts.PLUGIN_NAME)
        logger.setLevel(get_logging_level())
    return logger
```

-   **`get_logging_level()`**:
    -   **วัตถุประสงค์**: กำหนดระดับการ log (logging level) โดยอิงจากตัวแปรสภาพแวดล้อม `TORCH_PROFILER_LOG_LEVEL`
    -   **การทำงาน**:
        -   อ่านค่า `TORCH_PROFILER_LOG_LEVEL` จาก environment variable (ถ้าไม่มี, ใช้ 'INFO' เป็นค่าเริ่มต้น)
        -   แปลงค่าที่ได้เป็นตัวพิมพ์ใหญ่
        -   ตรวจสอบว่าค่าที่ได้เป็น logging level ที่ถูกต้องหรือไม่ (เช่น 'DEBUG', 'INFO', 'WARNING', 'ERROR')
        -   ถ้าไม่ถูกต้อง, จะใช้ `logging.INFO` เป็นค่าเริ่มต้น
    -   **คืนค่า**: Logging level (เช่น `logging.INFO`, `logging.DEBUG`)

-   **`get_logger()`**:
    -   **วัตถุประสงค์**: สร้างและคืนค่า instance ของ logger สำหรับปลั๊กอินนี้ (singleton pattern)
    -   **การทำงาน**:
        -   ใช้ `global logger` เพื่อให้ `logger` เป็น instance เดียวที่ใช้ร่วมกัน
        -   ถ้า `logger` ยังไม่ได้ถูกสร้าง (เป็น `None`):
            -   สร้าง logger ใหม่ด้วยชื่อที่กำหนดใน `consts.PLUGIN_NAME` (คือ `pytorch_profiler`)
            -   ตั้งค่า logging level โดยเรียก `get_logging_level()`
        -   คืนค่า `logger` instance
    -   **การใช้งาน**: ส่วนอื่นๆ ของโปรแกรมสามารถเรียก `utils.get_logger()` เพื่อรับ logger instance และใช้ในการบันทึกข้อความ log ต่างๆ

## ฟังก์ชันตรวจสอบประเภทไฟล์

```python
def is_chrome_trace_file(path):
    return consts.WORKER_PATTERN.match(path)
```

-   **`is_chrome_trace_file(path)`**:
    -   **วัตถุประสงค์**: ตรวจสอบว่าชื่อไฟล์ที่ระบุ (`path`) ตรงกับรูปแบบของไฟล์ Chrome Trace ที่ PyTorch Profiler สร้างขึ้นหรือไม่
    -   **การทำงาน**: ใช้ regular expression pattern ที่กำหนดไว้ใน `consts.WORKER_PATTERN` (จาก `consts.py`) เพื่อทดสอบกับ `path`
    -   **คืนค่า**: `True` ถ้าชื่อไฟล์ตรงกับ pattern, `False` ถ้าไม่ตรง
    -   **การใช้งาน**: ใช้ใน `TorchProfilerPlugin._get_run_dirs()` เพื่อค้นหาไดเรกทอรี run ที่มีไฟล์ trace

## ฟังก์ชันสร้าง Hyperlink

```python
def href(text, url):
    """"return html formatted hyperlink string

    Note:
        target="_blank" causes this link to be opened in new tab if clicked.
    """
    return f'<a href="{url}" target="_blank">{text}</a>'
```

-   **`href(text, url)`**:
    -   **วัตถุประสงค์**: สร้าง HTML string สำหรับ hyperlink
    -   **การทำงาน**: คืนค่า string ในรูปแบบ `<a href="URL" target="_blank">TEXT</a>`
        -   `target="_blank"`: ทำให้ลิงก์เปิดในแท็บใหม่เมื่อคลิก
    -   **การใช้งาน**: อาจใช้ในการสร้างลิงก์ในส่วนแสดงผลของ TensorBoard (แม้ว่าในโค้ดที่ให้มาจะยังไม่เห็นการใช้งานโดยตรง แต่เป็น utility ที่มีประโยชน์)

## คลาส `Canonicalizer`

```python
class Canonicalizer:
    def __init__(
            self,
            time_metric='us',
            memory_metric='B',
            *,
            input_time_metric='us',
            input_memory_metric='B'):
        # ... (รายละเอียดภายใน)

    def convert_time(self, t):
        return self.time_factor * t

    def convert_memory(self, m):
        return self.memory_factor * m
```

-   **`Canonicalizer`**:
    -   **วัตถุประสงค์**: ช่วยในการแปลงค่าเวลาและหน่วยความจำระหว่างหน่วยต่างๆ (เช่น จาก microsecond เป็น millisecond, หรือจาก Byte เป็น Megabyte)
    -   **`__init__(...)`**:
        -   รับ `time_metric` และ `memory_metric` ที่เป็นเป้าหมาย (output) และ `input_time_metric`, `input_memory_metric` ที่เป็นหน่วยของข้อมูลดิบ (input)
        -   **`time_metric_to_factor`**: Dictionary ที่ map ชื่อหน่วยเวลา ('us', 'ms', 's') ไปยังตัวคูณเพื่อแปลงเป็นหน่วยพื้นฐาน (microsecond)
        -   **`memory_metric_to_factor`**: Dictionary ที่ map ชื่อหน่วยความจำ ('B', 'KB', 'MB', 'GB') ไปยังตัวคูณเพื่อแปลงเป็นหน่วยพื้นฐาน (Byte)
        -   **`canonical_time_metrics` / `canonical_memory_metrics`**: Dictionaries สำหรับแปลงชื่อหน่วยที่ผู้ใช้อาจป้อนเข้ามาหลายรูปแบบ (เช่น 'micro', 'microsecond', 'us') ให้เป็นชื่อหน่วยมาตรฐาน ('us', 'ms', 's' หรือ 'B', 'KB', 'MB', 'GB')
        -   คำนวณ `self.time_factor` และ `self.memory_factor` ซึ่งเป็นอัตราส่วนสำหรับแปลงจากหน่วย input ไปยังหน่วย output ที่ต้องการ
            -   เช่น ถ้า input เป็น 'us' และ output เป็น 'ms', `time_factor` จะเป็น `1 / 1000 = 0.001`
    -   **`convert_time(self, t)`**: แปลงค่าเวลา `t` โดยคูณด้วย `self.time_factor`
    -   **`convert_memory(self, m)`**: แปลงค่าหน่วยความจำ `m` โดยคูณด้วย `self.memory_factor`
    -   **การใช้งาน**: ใช้ใน `RunProfile` (เช่น ใน `get_memory_stats`, `get_memory_curve`) เพื่อแสดงผลข้อมูลในหน่วยที่ผู้ใช้เลือกหรือหน่วยที่เหมาะสม

## คลาส `DisplayRounder`

```python
class DisplayRounder:
    def __init__(self, ndigits):
        self.ndigits = ndigits
        self.precision = pow(10, -ndigits)

    def __call__(self, v: float):
        _v = abs(v)
        if _v >= self.precision or v == 0:
            return round(v, 2) # หมายเหตุ: ในโค้ดจริงคือ round(v, self.ndigits) แต่ตัวอย่างใช้ 2
        else:
            ndigit = abs(math.floor(math.log10(_v)))
            return round(v, ndigit)
```

-   **`DisplayRounder`**:
    -   **วัตถุประสงค์**: ปัดเศษตัวเลขทศนิยมสำหรับการแสดงผล โดยพยายามแสดงตัวเลขที่มีนัยสำคัญ
    -   **`__init__(self, ndigits)`**:
        -   `ndigits`: จำนวนทศนิยมที่ต้องการเป็นหลัก (เช่น 2 ตำแหน่ง)
        -   `self.precision`: ค่าความละเอียดขั้นต่ำ (เช่น `10^-2 = 0.01`)
    -   **`__call__(self, v: float)`**:
        -   ถ้าค่าสัมบูรณ์ของ `v` มากกว่าหรือเท่ากับ `self.precision` หรือ `v` เป็น 0, จะปัดเศษ `v` ด้วยจำนวนทศนิยม `self.ndigits` (ในโค้ดตัวอย่างที่ให้มาเขียนเป็น `round(v, 2)` ซึ่งอาจจะไม่ตรงกับ `self.ndigits` เสมอไป ควรจะเป็น `round(v, self.ndigits)`)
        -   มิฉะนั้น (ถ้าค่าน้อยมาก), จะคำนวณจำนวนทศนิยมที่เหมาะสม (`ndigit`) โดยดูจากขนาดของ `v` (ใช้ `math.log10`) เพื่อให้แสดงตัวเลขที่มีนัยสำคัญตัวแรกๆ แล้วปัดเศษด้วยจำนวนทศนิยมนั้น
    -   **การใช้งาน**: ใช้ใน `RunProfile` เพื่อจัดรูปแบบตัวเลขก่อนส่งไปแสดงผล ทำให้ตัวเลขที่แสดงผลอ่านง่ายและไม่ยาวเกินไป

## Context Manager `timing`

```python
@contextmanager
def timing(description: str, force: bool = False) -> None:
    if force or os.environ.get('TORCH_PROFILER_BENCHMARK', '0') == '1':
        start = time.time()
        yield
        elapsed_time = time.time() - start
        logger.info(f'{description}: {elapsed_time}')
    else:
        yield
```

-   **`timing(description: str, force: bool = False)`**:
    -   **วัตถุประสงค์**: เป็น context manager สำหรับจับเวลาการทำงานของโค้ดบล็อกที่อยู่ภายใน `with` statement
    -   **การทำงาน**:
        -   ตรวจสอบว่าควรจะจับเวลาหรือไม่:
            -   ถ้า `force` เป็น `True` หรือ
            -   ถ้า environment variable `TORCH_PROFILER_BENCHMARK` ถูกตั้งค่าเป็น '1'
        -   ถ้าเงื่อนไขเป็นจริง:
            -   บันทึกเวลาเริ่มต้น (`start = time.time()`)
            -   `yield`: อนุญาตให้โค้ดบล็อกภายใน `with` ทำงาน
            -   เมื่อโค้ดบล็อกทำงานเสร็จ, คำนวณเวลาที่ผ่านไป (`elapsed_time`)
            -   Log ข้อความ `description` พร้อมกับ `elapsed_time` โดยใช้ `logger.info`
        -   ถ้าเงื่อนไขไม่เป็นจริง:
            -   `yield` โดยไม่ทำอะไรเพิ่มเติม (โค้ดบล็อกภายใน `with` ยังคงทำงานตามปกติ)
    -   **การใช้งาน**:
        ```python
        with utils.timing("Processing data"):
            # โค้ดที่ต้องการจับเวลา
            ...
        ```
        ช่วยในการทำ profiling หรือ benchmarking ส่วนต่างๆ ของโค้ดโปรไฟล์เลอร์เอง

## ฟังก์ชัน `lttb_sample` (Largest-Triangle-Three-Buckets)

```python
def _areas_of_triangles(a, bs, c):
    # ... (คำนวณพื้นที่สามเหลี่ยม) ...

def lttb_sample(memory_curves, n_out = 10240):
    # ... (โค้ดของ LTTB algorithm) ...
```

-   **`_areas_of_triangles(a, bs, c)`**:
    -   **วัตถุประสงค์**: Helper function สำหรับ `lttb_sample` ใช้คำนวณพื้นที่ของสามเหลี่ยมหลายๆ รูปพร้อมกัน โดยมีจุดยอดร่วม `a` และ `c` และจุดยอดที่สามมาจากลิสต์ `bs`
    -   **การทำงาน**: ใช้ NumPy broadcasting เพื่อคำนวณพื้นที่อย่างรวดเร็ว
-   **`lttb_sample(memory_curves, n_out = 10240)`**:
    -   **วัตถุประสงค์**: ลดจำนวนจุดข้อมูล (downsample) ใน `memory_curves` ให้เหลือประมาณ `n_out` จุด โดยใช้อัลกอริทึม Largest-Triangle-Three-Buckets (LTTB) ซึ่งพยายามรักษารูปร่างและลักษณะสำคัญของกราฟเดิมไว้ให้ได้มากที่สุด
    -   **`memory_curves`**: Dictionary ที่มี key เป็นชื่อ device (เช่น 'CPU', 'GPU0') และ value เป็นลิสต์ของจุดข้อมูล (เช่น `[[time1, allocated1, reserved1], [time2, allocated2, reserved2], ...]`)
    -   **`n_out`**: จำนวนจุดข้อมูลที่ต้องการหลังจากการ downsample (ค่าเริ่มต้น 10240)
    -   **การทำงานของ LTTB (โดยสังเขป)**:
        1.  ถ้าจำนวนจุดข้อมูลเดิมน้อยกว่าหรือเท่ากับ `n_out` จะไม่ทำอะไร
        2.  เก็บจุดแรกและจุดสุดท้ายของข้อมูลเดิมไว้เสมอ
        3.  แบ่งจุดข้อมูลที่เหลือ (ไม่รวมจุดแรกและจุดสุดท้าย) ออกเป็น `n_out - 2` "ถัง" (bins)
        4.  สำหรับแต่ละถัง:
            -   พิจารณาสามเหลี่ยมที่เกิดจาก:
                -   จุดที่ถูกเลือกไว้แล้วจากถังก่อนหน้า (A)
                -   แต่ละจุดในถังปัจจุบัน (B)
                -   จุดศูนย์กลาง (centroid) ของจุดทั้งหมดในถังถัดไป (C)
            -   เลือกจุด (B) จากถังปัจจุบันที่ทำให้เกิดสามเหลี่ยมที่มีพื้นที่ใหญ่ที่สุด จุดนี้จะถูกเก็บไว้เป็นตัวแทนของถังนี้
        5.  ทำซ้ำจนครบทุกถัง
    -   **คืนค่า**: Dictionary `sampled_memory_curves` ที่มีโครงสร้างเหมือน input แต่มีจำนวนจุดข้อมูลในแต่ละ curve ลดลง
    -   **การใช้งาน**: ใช้ใน `RunProfile.get_memory_curve()` เพื่อลดขนาดของข้อมูล memory curve ก่อนส่งไปแสดงผล ทำให้ frontend ทำงานได้เร็วขึ้นเมื่อมีข้อมูลจำนวนมาก

## สรุปฟังก์ชันอื่นๆ ที่อาจมี (จาก Context ของการใช้งานทั่วไปใน Profiler)

แม้ว่าโค้ดที่ให้มาจะไม่ได้แสดงฟังก์ชันทั้งหมดที่อาจอยู่ใน `utils.py` ของโปรไฟล์เลอร์ที่สมบูรณ์ แต่จากชื่อฟังก์ชันที่ผู้ใช้ระบุในคำถาม สามารถคาดเดาวัตถุประสงค์ได้ดังนี้:

-   **`get_gpu_id_from_trace_name(name)` / `get_trace_name_from_gpu_id(gpu_id)`**: แปลงระหว่างชื่อ trace (ที่อาจมี GPU ID) กับ GPU ID
-   **`get_resource_name(trace_name)`**: ดึงชื่อ resource (เช่น worker name) จากชื่อไฟล์ trace
-   **`is_memory_profile_file(path)` / `is_kernel_event_file(path)` / `is_tensorboard_trace_file(path)`**: ฟังก์ชันตรวจสอบประเภทไฟล์เพิ่มเติม คล้ายกับ `is_chrome_trace_file`
-   **`parse_ranges_str(ranges_str)`**: แยกวิเคราะห์สตริงที่ระบุช่วง (เช่น "0-5,7,10-12") ให้เป็นรายการของตัวเลขหรือช่วง
-   **`flatten_module_string(module_str)`**: อาจใช้ในการทำให้ชื่อโมดูลที่ซับซ้อน (มี hierarchy) กลายเป็นสตริงที่ง่ายขึ้น
-   **`is_valid_json(data)`**: ตรวจสอบว่าข้อมูลที่ให้มาเป็น JSON ที่ถูกต้องหรือไม่
-   **`round_float(value, precision)`**: ฟังก์ชันปัดเศษทศนิยม (อาจคล้ายกับ `DisplayRounder` หรือเป็นแบบง่ายกว่า)
-   **`format_time_us(value)` / `format_time_ms(value)` / `format_time_s(value)` / `format_time_str(value)`**: จัดรูปแบบค่าเวลาให้อยู่ในรูปสตริงที่อ่านง่าย พร้อมหน่วย (µs, ms, s)
-   **`format_size_str(value)` / `format_size_bytes(value)` / `format_size_gb(value)` / `format_size_mb(value)` / `format_size_kb(value)`**: จัดรูปแบบค่าขนาดหน่วยความจำให้อยู่ในรูปสตริงที่อ่านง่าย พร้อมหน่วย (B, KB, MB, GB)
-   **`format_percent_str(value)`**: จัดรูปแบบค่าเปอร์เซ็นต์ให้อยู่ในรูปสตริง
-   **`nsys_event_category_to_standard(category)`**: แปลง category ของ event จาก NVIDIA Nsight Systems (nsys) ให้เป็น category มาตรฐานที่โปรไฟล์เลอร์นี้ใช้
-   **`filter_stack_by_paths(stack_trace, paths_to_keep)`**: กรอง call stack ให้เหลือเฉพาะส่วนที่ตรงกับ path ที่สนใจ
-   **`get_stable_color_for_name(name)`**: สร้างสีที่ค่อนข้างคงที่สำหรับชื่อที่กำหนด (เช่น ชื่อ operator) เพื่อให้สีในกราฟมีความสอดคล้องกันระหว่างการ refresh
-   **`print_node_tree(node, indent='')`**: พิมพ์โครงสร้าง tree ของ `OperatorNode` (หรือ node อื่นๆ) ออกมาในรูปแบบที่อ่านง่าย (สำหรับ debugging)

ไฟล์ `utils.py` เป็นส่วนสำคัญที่ช่วยให้โค้ดส่วนอื่นๆ ของ `torch_tb_profiler` มีความกระชับ อ่านง่าย และสามารถนำฟังก์ชันที่ใช้บ่อยๆ กลับมาใช้ใหม่ได้โดยไม่ต้องเขียนซ้ำ
