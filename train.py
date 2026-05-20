from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

def main():
    datamodule = Folder(
        name="metal_nut",
        root="./datasets/mvtec/metal_nut",
        normal_dir="train/good",
        abnormal_dir="test/scratch",
        normal_test_dir="test/good",
        mask_dir="ground_truth/scratch",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,
    )

    model = Patchcore()
    engine = Engine()

    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()