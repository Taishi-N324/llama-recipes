import os
import json
import shutil
import argparse

def count_mds_files(base_dir):
    return sum(1 for filename in os.listdir(base_dir) if filename.endswith(".mds"))

def increment_filenames_and_update_index(base_dir, increment):
    index_path = os.path.join(base_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    new_filenames = {}

    for filename in sorted(os.listdir(base_dir), reverse=True):
        if filename.startswith("shard."):
            basename, extension = os.path.splitext(filename)
            prefix, number = basename.split(".")
            old_number = int(number)
            new_number = old_number + increment
            old_filename = f"{prefix}.{str(old_number).zfill(5)}{extension}"
            new_filename = f"{prefix}.{str(new_number).zfill(5)}{extension}"
            os.rename(os.path.join(base_dir, filename), os.path.join(base_dir, new_filename))
            new_filenames[old_filename] = new_filename

    for old_filename, new_filename in new_filenames.items():
        for shard in index["shards"]:
            print(old_filename, new_filename, shard["raw_data"]["basename"] )
            if shard["raw_data"]["basename"] == old_filename:
                shard["raw_data"]["basename"] = new_filename


    with open(index_path, "w") as f:
        print("index_path",index_path)
        json.dump(index, f)

def merge_indexes(dir1, dir2_updated):
    with open(os.path.join(dir1, "index.json"), "r") as f1, open(os.path.join(dir2_updated, "index.json"), "r") as f2:
        index1 = json.load(f1)
        index2 = json.load(f2)
    index1["shards"].extend(index2["shards"])
    with open(os.path.join(dir2_updated, "index.json"), "w") as f:
        json.dump(index1, f)

# Helper function to copy directory contents
def copy_directory_contents(src_directory, dest_directory):
    for item in os.listdir(src_directory):
        if item != "index.json":
            src = os.path.join(src_directory, item)
            dest = os.path.join(dest_directory, item)
            if os.path.isdir(src):
                shutil.copytree(src, dest, False, None)
            else:
                shutil.copy2(src, dest)


# Main function
def main(directories, dest):
    # Copy contents of first directory to dest
    shutil.copytree(directories[0], dest)

    # Process each directory
    for dir in directories[1:]:
        # Calculate increment value and update file name and index for each directory
        print(dir)
        increment_value = count_mds_files(dir)
        print("increment_value", increment_value)
        increment_filenames_and_update_index(dest, increment_value)
        merge_indexes(dir, dest)

        # Copy contents of each directory to dest except index.json
        copy_directory_contents(dir, dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs='+', required=True, help="Directories to process")
    parser.add_argument("--dest", required=True, help="Destination directory")
    args = parser.parse_args()
    main(args.dirs, args.dest)