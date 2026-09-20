"""Push a local LeRobot dataset to the Hugging Face Hub and file it in a collection.

    python examples/push_dataset.py --root datasets/lerobot/chess_mc \
        --repo-id XvKuoMing/so101_chess --collection so101_datasets

Runs in the VLA environment. Credentials come from the Hugging Face token
store (`huggingface-cli login`); no token is read from this repository.
"""
import argparse

from huggingface_hub import add_collection_item, create_collection, list_collections

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from chess_sim.hub import DATASET_REPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="local LeRobot dataset directory")
    ap.add_argument("--repo-id", default=DATASET_REPO, help="e.g. user/so101_chess")
    ap.add_argument("--collection", default=None, help="collection title to add the dataset to")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--tags", nargs="*", default=["robotics", "so101", "chess", "mujoco", "lerobot"])
    args = ap.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    print(f"pushing {dataset.meta.total_episodes} episodes / {dataset.meta.total_frames} frames "
          f"to {args.repo_id}")
    dataset.push_to_hub(tags=args.tags, private=args.private, upload_large_folder=True)
    print("pushed:", f"https://huggingface.co/datasets/{args.repo_id}")

    if args.collection:
        owner = args.repo_id.split("/")[0]
        existing = [c for c in list_collections(owner=owner, token=True) if c.title == args.collection]
        slug = existing[0].slug if existing else create_collection(
            args.collection, namespace=owner, exists_ok=True).slug
        add_collection_item(slug, item_id=args.repo_id, item_type="dataset", exists_ok=True)
        print(f"added to collection {args.collection}: https://huggingface.co/collections/{slug}")


if __name__ == "__main__":
    main()
