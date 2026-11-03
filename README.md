# Matrix Pipeline

MatrixPipeline is a modular, high-performance C++ computer vision application.
It utilizes a configurable Pipeline
Pattern that allows video frames (a.k.a. "matrix") flow through it.

This architecture creates a "T-junction" flow: lightweight tasks like rotation
and cropping run in real-time, while
heavy operations—such as YOLOv11 object detection, MP4 encoding, and network
I/O—are offloaded to isolated threads to
prevent frame drops. Key capabilities include multi-resolution MJPEG streaming,
motion-triggered recording, Matrix chat
alerts, and MQTT publishing, all defined via a flexible JSON configuration.

```mermaid
graph TD
    %% Define Styles
    classDef mainThread fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef asyncThread fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,stroke-dasharray: 5 5;
    classDef latThread fill:#ffebee,stroke:#c62828,stroke-width:2px,stroke-dasharray: 2 2;
    classDef io fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;

    %% Input
    RTSP([Camera RTSP Source]):::io --> Rotation

    %% ============================================================
    %% MAIN PIPELINE
    %% ============================================================
    subgraph Main_Pipeline [Main Processing Thread]
        direction TB
        Rotation(Rotate 270°) --> ChangeRate(Calc Change Rate<br/>5000ms)
        ChangeRate --> Ovl1(Overlay Info<br/>Generic)
        
        %% Flow continues to FPS Control
        Ovl1 --> FPS(Control FPS)
        FPS --> Crop(Crop Frame<br/>L:0 R:0.07 T:0.25 B:0.4)
        Crop --> OvlMain(Overlay Info<br/>Device Stats)
    end

    %% ============================================================
    %% ASYNC BRANCH 1: High Res View
    %% MOVED: Branches off from Ovl1 (Before FPS & Before Crop)
    %% ============================================================
    Ovl1 -.->|Clone & Push| Async1_Start(Async Queue):::asyncThread
    subgraph Async_Worker_1 [Async Thread 1: High Res View]
        direction TB
        Async1_Start --> Async1_Ovl(Overlay Info)
        Async1_Ovl --> HTTP_HR{{"HTTP Server :54320<br/>(High Res)"}}:::io
    end

    %% ============================================================
    %% ASYNC BRANCH 2: YOLOv11s (Port 54321)
    %% Branches from end of pipeline (Cropped & Low FPS)
    %% ============================================================
    OvlMain -.->|Clone & Push| Async2_Start(Async Queue):::asyncThread
    subgraph Async_Worker_2 [Async Thread 2: YOLOv11s]
        direction TB
        Async2_Start --> YOLO_S(Detect YOLOv11s<br/>Interval: 10)
        YOLO_S --> BBox_S(Overlay BBoxes)
        
        %% Latency Measure
        BBox_S --> LatStart_S(Latency Start):::latThread
        LatStart_S --> Ovl_S(Overlay Info<br/>Model: yolo11s)
        Ovl_S --> LatEnd_S(Latency End):::latThread

        %% Output
        LatEnd_S --> HTTP1{{HTTP Server :54321}}:::io
        HTTP1 --> Writer1(Video Writer<br/>.mp4):::io
        Writer1 --> Matrix1(Matrix Notifier):::io
    end

    %% ============================================================
    %% ASYNC BRANCH 3: YOLOv11m (Port 54322)
    %% ============================================================
    OvlMain -.->|Clone & Push| Async3_Start(Async Queue):::asyncThread
    subgraph Async_Worker_3 [Async Thread 3: YOLOv11m]
        direction TB
        Async3_Start --> YOLO_M(Detect YOLOv11m<br/>Interval: 10)
        YOLO_M --> BBox_M(Overlay BBoxes)
        
        %% Latency Measure
        BBox_M --> LatStart_M(Latency Start):::latThread
        LatStart_M --> Ovl_M(Overlay Info<br/>Model: yolo11m)
        Ovl_M --> LatEnd_M{{"Latency End<br/>P50/P90/P99"}}:::latThread

        %% Output
        LatEnd_M --> HTTP2{{HTTP Server :54322}}:::io
        HTTP2 --> Writer2(Video Writer<br/>.mp4):::io
        Writer2 --> Matrix2(Matrix Notifier):::io
    end

    %% Styling classes assignment
    class Rotation,ChangeRate,Ovl1,FPS,Crop,OvlMain mainThread;
    class YOLO_S,BBox_S,Ovl_S asyncThread;
    class YOLO_M,BBox_M,Ovl_M asyncThread;
    class Async1_Ovl asyncThread;
```

## Dependencies

- [vcpkg.json](./vcpkg.json)
- `v4l-utils`: for manually examining and manipulating local video devices.
- `OpenCV`: Check build notes [here](etc/build-notes.md)
  and [here](https://github.com/alex-lt-kong/the-nitty-gritty/tree/main/c-cpp/cpp-only/06_poc/05_cuda-vs-ffmpeg).
- [TensorRT](https://developer.nvidia.com/tensorrt/download)

## Build and deployment

```bash
mkdir ./build
cmake ../
make -j2
```

## Quality assurance

- The repo is tested with `Valgrind` from time to time:
  `valgrind --leak-check=yes --log-file=valgrind.rpt ./build/mp`.

### CTest

```bash
cmake -S tests -B build/tests -DCMAKE_TOOLCHAIN_FILE="$HOME/repos/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build/tests --parallel
ctest --test-dir build/tests --output-on-failure
```
