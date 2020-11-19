import parakeet

if __name__ == '__main__':
    import argparse
    import os
    import shutil
    from pathlib import Path
    
    package_path = Path(__file__).parent
    print(package_path)

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="cmd")
    
    list_exp_parser = subparser.add_parser("list-examples")
    clone = subparser.add_parser("clone-example")
    clone.add_argument("experiment_name", type=str, help="experiment name")
    
    args = parser.parse_args()
    
    if args.cmd == "list-examples":
        print(os.listdir(package_path / "examples"))
        exit(0)
    
    if args.cmd == "clone-example":
        source = package_path / "examples" / (args.experiment_name)
        target = Path(os.getcwd()) / (args.experiment_name)
        if not os.path.exists(str(source)):
            raise ValueError("{} does not exist".format(str(source)))
        
        if os.path.exists(str(target)):
            raise FileExistsError("{} already exists".format(str(target)))
        
        shutil.copytree(str(source), str(target))
        print("{} copied!".format(args.experiment_name))
        exit(0)
