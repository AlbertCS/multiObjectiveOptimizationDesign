import os

from pyrosetta import Pose, Vector1, init, rosetta


class Silent:
    """
    Methods for storing and retrieving poses from a silent file
    """

    def __init__(
        self,
        silent_file,
        params=None,
        patches=None,
        full_atom=True,
        overwrite=False,
        sugars=False,
    ):
        """
        Initialises the silent class using a path to the output silent file. If
        silent file already exists it will read its contents and append data to it.

        Parameters
        ==========
        silent_file : str
            Path to the output silent file.
        full_atom : bool
            Is this a full atom silent file?
        overwrite : bool
            Overwrites any previous silent file data.
        """

        init_options = ""
        if sugars:
            init_options = "-include_sugars "
            init_options += "-write_pdb_link_records "
            init_options += "-load_PDB_components false "
            init_options += "-alternate_3_letter_codes pdb_sugar "
            init_options += "-write_glycan_pdb_codes "
            init_options += "-auto_detect_glycan_connections "
            init_options += "-maintain_links "

        init(init_options)

        self.silent_file_path = silent_file

        # Initialise silent object
        self.silent_options = rosetta.core.io.silent.SilentFileOptions()
        self.silent_options.in_fullatom(full_atom)
        self.silent_file = rosetta.core.io.silent.SilentFileData(self.silent_options)
        self.silent_file.set_filename(self.silent_file_path)
        self.tags = []

        # Check params file input
        if isinstance(params, str):
            params = [params]
        self.params = params

        # Check patches file input
        if isinstance(patches, str):
            patches = [patches]
        self.patches = patches

        # Check if silent file already exists
        if os.path.exists(self.silent_file_path) and not overwrite:
            self.silent_file.read_file(self.silent_file_path)
            self.tags = list(self.silent_file.tags())

    def write_pose(self, pose, tag):
        """
        Writes a pose to the silent file using the given tag.

        Parameters
        ==========
        pose : pyrosetta.rosetta.core.pose.Pose
            Pose to write to the silent file.
        tag : str
            Name of the structure in the silent file.
        """

        if tag in self.tags:
            warning_message = (
                f"Tag {tag} already exists. Skipping writing of the given pose"
            )
            print(warning_message)
        else:
            silent_structure = rosetta.core.io.silent.BinarySilentStruct(
                self.silent_options, pose
            )
            silent_structure.set_decoy_tag(tag)
            self.silent_file.add_structure(silent_structure)
            self.silent_file.write_silent_struct(
                silent_structure, self.silent_file.filename()
            )
            self.tags.append(tag)

    def get_pose_from_tag(self, tag):
        """
        Get the pose associated with the given tag.

        Parameters
        =========
        tag : str
            Tag to search in the silent file.
        """

        # Check silent file
        if not self.tags:
            raise ValueError(
                f"No structures were found in silent file {self.silent_file.filename()}."
            )
        if tag not in self.tags:
            raise ValueError(
                f"No structure was found in silent file{self.silent_file.filename()} with the tag {tag}."
            )

        # Get pose from silent file and assign it to an empty pose
        ss = self.silent_file.get_structure(tag)
        pose = Pose()
        if self.params:
            params = Vector1(self.params)
            res_set = pose.conformation().modifiable_residue_type_set_for_conf()
            res_set.read_files_for_base_residue_types(params)

            if self.patches:
                patches = rosetta.utility.vector1_std_string(0)
                meta_patches = rosetta.utility.vector1_std_string(0)
                for patch_file in self.patches:
                    patches.append(patch_file)
                res_set.add_patches(patches, meta_patches)
            pose.conformation().reset_residue_type_set_for_conf(res_set)

        ss.fill_pose(pose)

        return pose

    def get_tags(self):
        """
        Returns a list of tags in the silent file.
        """

        return self.tags

    def read_poses_from_silent_file(self):
        """
        Reads all the poses from the silent file and returns a list of poses.
        """

        poses = []
        for tag in self.tags:
            poses.append(self.get_pose_from_tag(tag))

        return poses

    def extract_poses(self, output_folder, remove_last_index=True, overwrite=False):
        """
        Extract all poses to the specified folder
        """

        # Create the output folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Extract poses to PDB
        for tag in sorted(self.tags):

            if remove_last_index:
                output_file = (
                    output_folder + "/" + "_".join(tag.split("_")[:-1]) + ".pdb"
                )
            else:
                output_file = output_folder + "/" + tag + ".pdb"

            if not os.path.exists(output_file) or overwrite:
                pose = self.get_pose_from_tag(tag)
                pose.dump_pdb(output_file)
