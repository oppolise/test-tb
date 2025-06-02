# เอกสารอธิบายไฟล์ plugin.py

ไฟล์ `plugin.py` เป็นหัวใจหลักของ `torch_tb_profiler` ที่ทำหน้าที่เป็นปลั๊กอินสำหรับ TensorBoard โดยตรง ไฟล์นี้ประกอบด้วยคลาส `TorchProfilerPlugin` ซึ่งสืบทอดมาจาก `base_plugin.TBPlugin` ของ TensorBoard และจัดการการทำงานทั้งหมดตั้งแต่การโหลดข้อมูลโปรไฟล์, การให้บริการ API สำหรับ frontend, ไปจนถึงการจัดการ run และ worker ต่างๆ

## ภาพรวม

`TorchProfilerPlugin` รับผิดชอบงานหลักๆ ดังนี้:

1.  **การจัดการ Log Directory**: ตรวจสอบและกำหนด log directory ที่จะใช้ในการค้นหาข้อมูลโปรไฟล์
2.  **การค้นหาและโหลด Runs**: มี thread แยก (`_monitor_runs`) คอยสแกน log directory เพื่อค้นหา run ใหม่ๆ และเมื่อพบ run ใหม่ จะมี thread (`_load_run`) แยกอีกชุดหนึ่งทำการโหลดข้อมูลของ run นั้นๆ (ไฟล์ trace)
3.  **การจัดการข้อมูลโปรไฟล์**:
    *   ใช้ `RunLoader` ในการแยกวิเคราะห์ไฟล์ trace (เช่น `.pt.trace.json`)
    *   เก็บข้อมูล run ที่โหลดแล้วใน `_runs` (เป็น `OrderedDict`)
    *   ใช้ `_cache` (เป็น `io.Cache`) สำหรับการจัดเก็บข้อมูลชั่วคราวหรือข้อมูลที่ประมวลผลแล้ว เพื่อลดเวลาในการโหลดซ้ำ
    *   มี `_queue` สำหรับการสื่อสารระหว่าง thread ที่ค้นหา run และ thread ที่รับ run เข้ามาในระบบ
4.  **การให้บริการ API (Routes)**: กำหนด routes ต่างๆ ที่ frontend ของ TensorBoard จะเรียกใช้เพื่อขอข้อมูล เช่น รายการ runs, รายการ views, ข้อมูล overview, ข้อมูล operation, ข้อมูล kernel, trace data, และข้อมูลอื่นๆ ที่เกี่ยวข้องกับการแสดงผลโปรไฟล์
5.  **การจัดการสถานะ**: มี `is_active()` เพื่อบอก TensorBoard ว่าปลั๊กอินนี้มีข้อมูลที่พร้อมจะแสดงผลหรือไม่
6.  **การทำความสะอาด**: ใช้ `atexit.register(clean)` เพื่อลบ temporary directory (`_temp_dir`) เมื่อโปรแกรมปิดตัวลง

## การนำเข้าโมดูล (Import Modules)

ไฟล์นี้มีการนำเข้าโมดูลที่จำเป็นหลายอย่าง:

-   `atexit`: สำหรับการลงทะเบียนฟังก์ชันที่จะถูกเรียกเมื่อโปรแกรม Python สิ้นสุดการทำงาน
-   `gzip`: สำหรับการบีบอัดและคลายการบีบอัดข้อมูล (เช่น ไฟล์ trace ที่อาจเป็น `.gz`)
-   `json`: สำหรับการทำงานกับข้อมูลในรูปแบบ JSON
-   `os`, `shutil`, `sys`, `tempfile`: สำหรับการจัดการไฟล์, ไดเรกทอรี, และการโต้ตอบกับระบบปฏิบัติการ
-   `threading`, `time`, `collections.OrderedDict`, `queue.Queue`: สำหรับการทำงานแบบ multi-threading, การจัดการเวลา, โครงสร้างข้อมูล, และคิว
-   `werkzeug`: สำหรับการจัดการ HTTP requests และ responses ซึ่งเป็นส่วนหนึ่งของ WSGI (Web Server Gateway Interface) ที่ TensorBoard ใช้
-   `tensorboard.plugins.base_plugin`: ซึ่งมีคลาสแม่ `TBPlugin` ที่ `TorchProfilerPlugin` สืบทอดมา
-   `. import consts, io, utils`: นำเข้าโมดูลภายในโปรเจกต์เอง เช่น ค่าคงที่, ฟังก์ชัน I/O, และ utilities
-   `.profiler import RunLoader`: สำหรับการโหลดและประมวลผล run
-   `.run import DistributedRunProfile, Run, RunProfile`: คลาสที่ใช้แทน run และ profile ของ run นั้นๆ

## ฟังก์ชัน `decorate_headers`

```python
def decorate_headers(func):
    def wrapper(*args, **kwargs):
        headers = func(*args, **kwargs)
        headers.extend(TorchProfilerPlugin.headers)
        return headers
    return wrapper

exceptions.HTTPException.get_headers = decorate_headers(exceptions.HTTPException.get_headers)
```

-   **วัตถุประสงค์**: ฟังก์ชันนี้เป็น decorator ที่ใช้ในการเพิ่ม HTTP headers ที่กำหนดไว้ใน `TorchProfilerPlugin.headers` (คือ `('X-Content-Type-Options', 'nosniff')`) เข้าไปในทุกๆ HTTP response ที่เกิดจาก `exceptions.HTTPException`
-   **การทำงาน**:
    -   `decorate_headers` รับฟังก์ชัน `func` (ซึ่งในที่นี้คือ `exceptions.HTTPException.get_headers` เดิม) เป็นอาร์กิวเมนต์
    -   `wrapper` จะเรียก `func` เดิมเพื่อรับ headers เริ่มต้น จากนั้นเพิ่ม `TorchProfilerPlugin.headers` เข้าไป
    -   `exceptions.HTTPException.get_headers = ...`: เป็นการ "monkey patch" เมธอด `get_headers` ของคลาส `HTTPException` ใน `werkzeug.exceptions` เพื่อให้ทุก exception response มี header `X-Content-Type-Options: nosniff` ซึ่งเป็นมาตรการความปลอดภัยเพื่อป้องกันเบราว์เซอร์จากการพยายาม "เดา" content type ของไฟล์เอง
-   **ทำไมต้องกำหนด**: เพื่อเพิ่มความปลอดภัยให้กับ HTTP responses ของปลั๊กอิน

## คลาส `TorchProfilerPlugin`

นี่คือคลาสหลักของปลั๊กอิน

```python
class TorchProfilerPlugin(base_plugin.TBPlugin):
    plugin_name = consts.PLUGIN_NAME
    headers = [('X-Content-Type-Options', 'nosniff')]
    CONTENT_TYPE = 'application/json'
```

-   `plugin_name = consts.PLUGIN_NAME`: กำหนดชื่อของปลั๊กอินนี้ ซึ่งจะถูกใช้โดย TensorBoard เพื่อระบุและโหลดปลั๊กอิน (ค่ามาจาก `consts.py`)
-   `headers`: รายการของ HTTP headers ที่จะถูกเพิ่มเข้าไปใน response ต่างๆ (ดังที่เห็นใน `decorate_headers` และ `static_file_route`, `respond_as_json`)
-   `CONTENT_TYPE = 'application/json'`: กำหนด content type เริ่มต้นสำหรับ JSON responses

### `__init__(self, context: base_plugin.TBContext)`

-   **วัตถุประสงค์**: Constructor ของคลาส ใช้สำหรับ khởi tạo instance ของปลั๊กอิน
-   **การทำงาน**:
    -   เรียก constructor ของคลาสแม่ (`super().__init__(context)`)
    -   **กำหนด `logdir`**:
        -   ตรวจสอบว่า `context.logdir` (ไดเรกทอรี log หลักที่ TensorBoard ใช้) มีค่าหรือไม่
        -   ถ้าไม่มี แต่มี `context.flags.logdir_spec` (ซึ่งอาจระบุหลายไดเรกทอรีคั่นด้วยจุลภาค), จะใช้ไดเรกทอรีแรกที่ระบุใน `logdir_spec` และแสดงคำเตือนถ้ามีหลายไดเรกทอรี
        -   ใช้ `io.abspath()` เพื่อให้ได้ path แบบเต็ม และ `rstrip('/')` เพื่อลบ `/` ที่อาจอยู่ท้ายสุด
    -   **Locks**:
        -   `_load_lock = threading.Lock()`: Lock สำหรับป้องกัน race condition ในการเข้าถึง `_load_threads`
        -   `_runs_lock = threading.Lock()`: Lock สำหรับป้องกัน race condition ในการเข้าถึง `_runs`
    -   `_load_threads = []`: ลิสต์สำหรับเก็บ thread ที่กำลังทำงานโหลด run อยู่
    -   `_runs = OrderedDict()`: OrderedDict สำหรับเก็บ instance ของ `Run` ที่โหลดแล้ว โดยใช้ชื่อ run เป็น key (OrderedDict ช่วยให้ลำดับของ run ที่แสดงผลสอดคล้องกับลำดับที่เพิ่มเข้ามาหรือเรียงตามชื่อ)
    -   `_temp_dir = tempfile.mkdtemp()`: สร้าง temporary directory สำหรับเก็บไฟล์ชั่วคราว เช่น cache หรือไฟล์ trace ที่มีการแก้ไข
    -   `_cache = io.Cache(self._temp_dir)`: สร้าง instance ของ `io.Cache` เพื่อจัดการการ caching ข้อมูลลงใน `_temp_dir`
    -   `_queue = Queue()`: สร้าง queue สำหรับการสื่อสารระหว่าง thread `_monitor_runs` และ `_receive_runs`
    -   `_gpu_metrics_file_dict = {}`: Dictionary สำหรับเก็บ mapping ระหว่าง path ของไฟล์ trace ดั้งเดิม กับ path ของไฟล์ trace ชั่วคราวที่ถูกเพิ่มข้อมูล GPU metrics เข้าไปแล้ว (ใช้ใน `trace_route`)
    -   **เริ่ม Threads**:
        -   `monitor_runs = threading.Thread(target=self._monitor_runs, ...)`: สร้างและเริ่ม thread ชื่อ `monitor_runs` ซึ่งจะคอยสแกนหา run ใหม่ๆ ใน `logdir`
        -   `receive_runs = threading.Thread(target=self._receive_runs, ...)`: สร้างและเริ่ม thread ชื่อ `receive_runs` ซึ่งจะคอยรับ run ที่โหลดเสร็จแล้วจาก `_queue` และเพิ่มเข้าไปใน `_runs`
    -   `diff_run_cache = {}`, `diff_run_flatten_cache = {}`: Cache สำหรับเก็บผลลัพธ์ของการเปรียบเทียบ run (diff) เพื่อลดการคำนวณซ้ำ
    -   **ลงทะเบียน `clean` function**:
        -   `atexit.register(clean)`: ลงทะเบียนฟังก์ชัน `clean` ให้ถูกเรียกเมื่อโปรแกรม Python ปิดตัวลง ฟังก์ชัน `clean` นี้จะทำการล้าง `_cache` และลบ `_temp_dir`

### `is_active(self)`

-   **วัตถุประสงค์**: เมธอดนี้ถูกเรียกโดย TensorBoard เพื่อตรวจสอบว่าปลั๊กอินนี้ควรจะแสดงใน UI หรือไม่
-   **การทำงาน**:
    -   ถ้า `self.is_loading` เป็น `True` (หมายถึงกำลังมี thread โหลด run อยู่) จะคืนค่า `True`
    -   มิฉะนั้น จะคืนค่า `True` ถ้า `_runs` (ลิสต์ของ run ที่โหลดแล้ว) ไม่ว่าง และ `False` ถ้าว่าง
-   **ผลลัพธ์**: ถ้าคืนค่า `True`, TensorBoard จะแสดงแท็บของปลั๊กอินนี้

### `get_plugin_apps(self)`

-   **วัตถุประสงค์**: เมธอดนี้คืนค่า dictionary ที่ map URL paths (routes) ไปยัง handler functions ที่จะจัดการ request ที่เข้ามายัง path เหล่านั้น
-   **การทำงาน**: คืนค่า dictionary ที่มี key เป็น URL path (เช่น `/runs`, `/overview`) และ value เป็นเมธอดของคลาสที่จะจัดการ request นั้น (เช่น `self.runs_route`, `self.overview_route`)
-   **Routes ที่สำคัญ**:
    -   `/index.js`, `/index.html`, `/trace_viewer_full.html`, `/trace_embedding.html`: ให้บริการไฟล์ static (JavaScript, HTML) สำหรับ frontend
    -   `/runs`: คืนค่ารายการ run ทั้งหมด
    -   `/views`: คืนค่ารายการ view ที่มีสำหรับ run ที่ระบุ
    -   `/workers`: คืนค่ารายการ worker สำหรับ run และ view ที่ระบุ
    -   `/spans`: คืนค่ารายการ span (ช่วงเวลา) สำหรับ run และ worker ที่ระบุ (ถ้ามี)
    -   `/overview`: คืนค่าข้อมูลภาพรวมของ profile
    -   `/operation`, `/operation/table`, `/operation/stack`: คืนค่าข้อมูลเกี่ยวกับ operations (pie chart, table, call stack)
    -   `/kernel`, `/kernel/table`, `/kernel/tc_pie`: คืนค่าข้อมูลเกี่ยวกับ kernels (pie chart, table, Tensor Core usage)
    -   `/trace`: คืนค่าข้อมูล trace ในรูปแบบ JSON (อาจมีการเพิ่ม GPU metrics)
    -   `/distributed/*`: Routes สำหรับข้อมูล distributed training (GPU info, overlap, wait time, communication ops)
    -   `/memory`, `/memory_curve`, `/memory_events`: Routes สำหรับข้อมูลการใช้หน่วยความจำ
    -   `/module`, `/tree`: Routes สำหรับข้อมูล module view และ operator tree view
    -   `/diff`, `/diffnode`: Routes สำหรับการเปรียบเทียบ (diff) ระหว่างสอง run/profile

### `frontend_metadata(self)`

-   **วัตถุประสงค์**: คืนค่า metadata เกี่ยวกับ frontend ของปลั๊กอิน
-   **การทำงาน**: คืนค่า `base_plugin.FrontendMetadata` object ซึ่งระบุ path ไปยัง ES module หลักของ frontend (`/index.js`) และตั้งค่า `disable_reload=True` (หมายความว่า frontend จะไม่ถูกโหลดใหม่เมื่อ TensorBoard refresh ข้อมูล)

### Routes (เช่น `runs_route`, `overview_route`, ฯลฯ)

แต่ละเมธอดที่เป็น route handler จะมี `@wrappers.Request.application` decorator ซึ่งหมายความว่าเมธอดนั้นเป็น WSGI application ที่สามารถจัดการ `werkzeug.Request` object ได้

-   **การทำงานทั่วไปของ Route Handlers**:
    1.  **รับ Parameters**: ดึงค่าพารามิเตอร์จาก request URL (เช่น `run_name = request.args.get('run')`)
    2.  **Validate Parameters**: ใช้ `_validate()` เพื่อตรวจสอบว่าพารามิเตอร์ที่จำเป็นมีค่าครบถ้วนหรือไม่ ถ้าไม่ จะ raise `exceptions.BadRequest`
    3.  **รับ Run/Profile**:
        -   ใช้ `_get_run(name)` เพื่อดึง instance ของ `Run`
        -   ใช้ `_get_profile(name, worker, span)` หรือ `_get_profile_for_request()` หรือ `_get_distributed_profile_for_request()` เพื่อดึง instance ของ `RunProfile` หรือ `DistributedRunProfile` ที่เหมาะสม
        -   ถ้าไม่พบ run หรือ profile จะ raise `exceptions.NotFound`
    4.  **ประมวลผลข้อมูล**: เรียกเมธอดจาก `Run` หรือ `RunProfile` object เพื่อดึงข้อมูลที่ต้องการ (เช่น `profile.overview`, `profile.operation_pie_by_name`)
    5.  **ตอบกลับเป็น JSON**: ใช้ `self.respond_as_json(data, compress)` เพื่อแปลง Python dictionary/list เป็น JSON string และส่งกลับเป็น HTTP response พร้อมกับ `Content-Type: application/json` และ headers ที่กำหนดไว้ อาจมีการบีบอัดข้อมูลถ้า `compress=True`

-   **ตัวอย่าง Route เฉพาะ**:
    -   `runs_route`: คืนค่ารายชื่อ run ที่มี และสถานะการโหลด (`self.is_loading`)
    -   `views_route`: คืนค่ารายการ display name ของ view ที่มีใน run ที่ระบุ
    -   `trace_route`: มีความซับซ้อนเล็กน้อย:
        -   ถ้าเป็น pure CPU trace และไม่ใช่ `.gz` จะบีบอัดข้อมูลก่อนส่ง
        -   ถ้ามี GPU metrics:
            -   ตรวจสอบว่าเคยมีการสร้างไฟล์ trace ที่รวม GPU metrics แล้วหรือยัง (ใน `_gpu_metrics_file_dict`)
            -   ถ้ามีแล้ว ให้อ่านจากไฟล์นั้น
            -   ถ้ายังไม่มี ให้อ่านไฟล์ trace ดั้งเดิม, คลายการบีบอัดถ้าจำเป็น (`.gz`), เรียก `profile.append_gpu_metrics(raw_data)` เพื่อเพิ่มข้อมูล GPU, จากนั้นเขียนข้อมูลที่รวมแล้วลงในไฟล์ชั่วคราว (เป็น `.json.gz`) และเก็บ path ของไฟล์นี้ไว้ใน `_gpu_metrics_file_dict`
        -   ส่งข้อมูล trace กลับพร้อม header `Content-Encoding: gzip`

### `static_file_route(self, request: werkzeug.Request)`

-   **วัตถุประสงค์**: ให้บริการไฟล์ static (HTML, CSS, JavaScript) ที่ใช้สำหรับ frontend ของปลั๊กอิน
-   **การทำงาน**:
    -   ดึงชื่อไฟล์จาก `request.path`
    -   กำหนด MIME type ตามนามสกุลไฟล์
    -   สร้าง path เต็มไปยังไฟล์ในไดเรกทอรี `static` (ซึ่งควรจะอยู่ในไดเรกทอรีเดียวกับ `plugin.py`)
    -   อ่านเนื้อหาไฟล์และส่งกลับเป็น `werkzeug.Response` พร้อม MIME type และ headers ที่ถูกต้อง
    -   ถ้าไม่พบไฟล์ จะ raise `exceptions.NotFound`

### `respond_as_json(obj, compress: bool = False)`

-   **วัตถุประสงค์**: เป็น static method (แต่ในโค้ดไม่ได้ประกาศเป็น `@staticmethod` ซึ่งก็ยังทำงานได้) ที่ช่วยแปลง Python object (เช่น dict, list) เป็น JSON HTTP response
-   **การทำงาน**:
    -   แปลง `obj` เป็น JSON string
    -   ถ้า `compress` เป็น `True`, จะบีบอัด JSON string นั้นด้วย gzip และเพิ่ม header `Content-Encoding: gzip`
    -   คืนค่า `werkzeug.Response`

### Properties และ Helper Methods ภายใน

-   `is_loading` (property):
    -   คืนค่า `True` ถ้า `_load_threads` (ลิสต์ของ thread ที่กำลังโหลด run) ไม่ว่าง, `False` ถ้าว่าง
    -   ใช้ `_load_lock` เพื่อความปลอดภัยของ thread

-   `get_diff_runs(self, request: werkzeug.Request)`:
    -   ดึงชื่อ run, worker, span สำหรับ "base" run และ "exp" (experiment) run จาก request
    -   ตรวจสอบความถูกต้องของพารามิเตอร์
    -   คืนค่า tuple ของ `(base_profile, exp_profile)`

-   `get_diff_status(self, base: RunProfile, exp: RunProfile)`:
    -   ใช้ `(base, exp)` เป็น key ในการค้นหาจาก `self.diff_run_cache`
    -   ถ้าไม่พบ, จะเรียก `base.compare_run(exp)` เพื่อคำนวณ diff statistics, เก็บผลลัพธ์ไว้ใน cache, แล้วคืนค่า
    -   ถ้าพบใน cache, คืนค่าจาก cache

-   `get_diff_stats_dict(self, base: RunProfile, exp: RunProfile)`:
    -   คล้ายกับ `get_diff_status` แต่ใช้ `self.diff_run_flatten_cache`
    -   ถ้าไม่พบใน cache, จะเรียก `diff_stats.flatten_diff_tree()` เพื่อสร้าง dictionary ของ diff statistics, เก็บผลลัพธ์ไว้ใน cache, แล้วคืนค่า

-   `_monitor_runs(self)`:
    -   ทำงานใน background thread
    -   วนลูปไม่รู้จบ (daemon thread จะถูกปิดเมื่อ main thread จบ)
    -   ในแต่ละรอบ:
        -   เรียก `_get_run_dirs()` เพื่อสแกนหาไดเรกทอรี run ใน `self.logdir`
        -   สำหรับแต่ละ run directory ที่พบและยังไม่เคย "แตะ" (`touched`):
            -   เพิ่มเข้า `touched` set
            -   สร้าง thread ใหม่ (`_load_run`) เพื่อโหลด run นั้นๆ และเพิ่ม thread นี้เข้า `_load_threads`
        -   ถ้าไม่มี run directory เลย (เช่น ถูกลบไป), จะเคลียร์ `_runs`
        -   จัดการ exception ที่อาจเกิดขึ้นระหว่างการสแกน
        -   `time.sleep(consts.MONITOR_RUN_REFRESH_INTERNAL_IN_SECONDS)` เพื่อรอสักครู่ก่อนสแกนรอบถัดไป

-   `_receive_runs(self)`:
    -   ทำงานใน background thread
    -   วนลูปไม่รู้จบ, รอรับ `Run` object จาก `self._queue` (ซึ่งถูก `put` โดย `_load_run` เมื่อโหลดเสร็จ)
    -   เมื่อได้รับ `Run` object:
        -   เพิ่ม/อัปเดต run นั้นใน `self._runs` (ใช้ `_runs_lock`)
        -   ถ้าเป็น run ใหม่, จะเรียง `_runs` ใหม่ตามชื่อ run (เนื่องจาก `_runs` เป็น `OrderedDict`)

-   `_get_run_dirs(self)`:
    -   สแกน `self.logdir`
    -   `yield` path ของไดเรกทอรีที่ถือว่าเป็น run (คือ ไดเรกทอรีที่มีไฟล์ `.pt.trace.json` หรือ `.pt.trace.json.gz` อย่างน้อยหนึ่งไฟล์)

-   `_load_run(self, run_dir)`:
    -   ถูกเรียกใน thread แยกสำหรับแต่ละ run ที่จะโหลด
    -   คำนวณชื่อ run จาก `run_dir` และ `self.logdir` (ด้วย `_get_run_name`)
    -   สร้าง `RunLoader` instance และเรียก `loader.load()` เพื่อโหลดข้อมูล run (ซึ่งจะอ่านไฟล์ trace, ประมวลผล, และสร้าง `Run` object)
    -   เมื่อโหลดเสร็จ, ใส่ `Run` object ที่ได้เข้าไปใน `self._queue` เพื่อให้ `_receive_runs` จัดการต่อไป
    -   จัดการ exception ที่อาจเกิดขึ้นระหว่างการโหลด
    -   เมื่อเสร็จสิ้น (ไม่ว่าจะสำเร็จหรือล้มเหลว), จะลบ thread ปัจจุบันออกจาก `_load_threads` (ใช้ `_load_lock`)

-   `_get_run(self, name) -> Run`:
    -   ดึง `Run` object จาก `self._runs` ตามชื่อที่ระบุ (ใช้ `_runs_lock`)
    -   ถ้าไม่พบ, raise `exceptions.NotFound`

-   `_get_run_name(self, run_dir)`:
    -   คำนวณชื่อ run ที่จะแสดงใน UI โดยอิงจาก path ของ `run_dir` เทียบกับ `self.logdir`
    -   ถ้า `run_dir` คือ `logdir` เอง, ชื่อ run จะเป็นชื่อของ `logdir`
    -   มิฉะนั้น, ชื่อ run จะเป็น relative path จาก `logdir` ไปยัง `run_dir`

-   `_get_profile_for_request(self, request: werkzeug.Request) -> RunProfile`:
    -   Helper method สำหรับดึง `RunProfile` จาก request โดยดึง 'run', 'worker', 'span' จาก request args
    -   ตรวจสอบว่า profile ที่ได้เป็น instance ของ `RunProfile` จริงๆ

-   `_get_distributed_profile_for_request(self, request: werkzeug.Request) -> DistributedRunProfile`:
    -   คล้ายกับด้านบน แต่สำหรับ distributed view โดยจะใช้ worker 'All' เสมอ
    -   ตรวจสอบว่า profile ที่ได้เป็น instance ของ `DistributedRunProfile` จริงๆ

-   `_get_profile(self, name, worker, span)`:
    -   ดึง `Run` object ด้วย `_get_run(name)`
    -   เรียก `run.get_profile(worker, span)` เพื่อเอา profile ที่ต้องการจาก run นั้น
    -   ถ้าไม่พบ profile, raise `exceptions.NotFound`

-   `_validate(self, **kwargs)`:
    -   ตรวจสอบว่าพารามิเตอร์ที่ส่งมาใน `kwargs` (ซึ่งควรจะเป็นพารามิเตอร์จาก request URL) มีค่า (`is not None`)
    -   ถ้าพารามิเตอร์ใดมีค่าเป็น `None`, raise `exceptions.BadRequest` พร้อมระบุชื่อพารามิเตอร์ที่ขาดไป

## สรุป

`plugin.py` เป็นแกนกลางที่ซับซ้อนแต่มีโครงสร้างที่ดีสำหรับการทำงานของ PyTorch Profiler plugin ใน TensorBoard มันจัดการวงจรชีวิตของข้อมูลโปรไฟล์ตั้งแต่การค้นพบ, การโหลด, การประมวลผล, การ caching, ไปจนถึงการให้บริการข้อมูลนั้นๆ แก่ frontend ผ่านทาง HTTP API ที่กำหนดไว้อย่างชัดเจน การใช้ multi-threading ช่วยให้การโหลดข้อมูลและการตอบสนองของ UI เป็นไปอย่างราบรื่น
