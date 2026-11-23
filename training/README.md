# Training Jobs

Containerized training jobs for Agent Platform Custom Jobs.

| Job | Path | Description |
|-----|------|-------------|
| ALS + Ranker | [`als/`](als/) | ALS retrieval + ALS-specific LightGBM LambdaMART ranker |
| Two-Tower + Ranker | [`two_tower/`](two_tower/) | Pure ID-only dual encoder + Two-Tower LambdaMART ranker |
| Hybrid + Ranker | [`hybrid/`](hybrid/) | Feature-rich dual encoder + Hybrid LambdaMART ranker |
| Distributed ALS (Dataproc Serverless) | [`distributed/dataproc/`](distributed/dataproc/) | Spark MLlib ALS + sharded eval via Serverless batch |

See each subdirectory's README for build, push, and run instructions.
