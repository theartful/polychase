import json
import textwrap

import bpy
import bpy.props

from . import dependency_utils


class AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    visible_preferences: bpy.props.EnumProperty(
        name="Visible Preferences",
        items=(("DEPENDENCIES", "Dependencies", ""),),
    )

    sys_path_list_str: bpy.props.StringProperty(name="System Path List Decoded String", default="[]")

    @classmethod
    def register(cls):
        """Register corresponding operators."""
        bpy.utils.register_class(dependency_utils.InstallDependenciesOperator)
        bpy.utils.register_class(dependency_utils.UninstallDependenciesOperator)

    @classmethod
    def unregister(cls):
        """Register corresponding operators."""
        bpy.utils.unregister_class(dependency_utils.InstallDependenciesOperator)
        bpy.utils.unregister_class(dependency_utils.UninstallDependenciesOperator)

    @staticmethod
    def _get_installation_status_str(installation_status):
        if installation_status:
            status = "Installed"
        else:
            status = "Not installed"
        return status

    @staticmethod
    def _get_package_info_str(
        info_str,
        installation_status,
        setuptools_missing_str="",
        removed_in_current_sesssion_str="",
    ):
        if installation_status:
            if info_str is None:
                # In this case the info_str could not been determined, since
                # the setuptools package is missing
                info_str = setuptools_missing_str
        else:
            if info_str is None:
                # In this case the module has not been removed in the
                # current Blender session
                info_str = ""
            else:
                # In this case the module was previously installed and has
                # been removed in the current Blender session
                info_str = removed_in_current_sesssion_str
        return info_str

    def draw(self, context):
        """Draw available preference options."""
        layout = self.layout

        layout.row().prop(self, "visible_preferences", expand=True)
        layout.row()
        if self.visible_preferences == "DEPENDENCIES":
            install_dependency_box = layout.box()
            self._draw_dependencies(install_dependency_box)

    def _draw_dependency(
        self,
        dependency,
        setuptools_missing_str,
        removed_in_current_sesssion_str,
        dependencies_status_box,
    ):
        dependency_installation_status = dependency.installation_status
        dependency_installation_status_str = self._get_installation_status_str(dependency_installation_status)
        version_str, location_str = dependency.get_package_info()
        version_str = self._get_package_info_str(
            version_str,
            dependency_installation_status,
            setuptools_missing_str,
            removed_in_current_sesssion_str,
        )
        location_str = self._get_package_info_str(location_str, dependency_installation_status)

        dependency_status_box = dependencies_status_box.box()

        # https://blender.stackexchange.com/questions/51256/how-to-create-uilist-with-auto-aligned-three-columns
        #  This layout approach splits the remainder of the row (recursively).
        name_column = dependency_status_box.split(factor=0.1)
        name_column.label(text=f"{dependency.gui_name}")
        status_column = name_column.split(factor=0.1)
        status_column.label(text=f"{dependency_installation_status_str}")
        verson_location_column = status_column.split()
        version_location_str = f"{version_str}"
        if location_str != "":
            # Blender does not support tabs "\t"
            version_location_str += f"    ({location_str})"
        verson_location_column.label(text=version_location_str)

        # This layout approach adds more columns than required to enforce left
        #  alignment of buttons.
        column_flow_layout = dependency_status_box.column_flow(columns=5)
        package_name = dependency.package_name
        # Add operator to first column
        install_dependency_props = column_flow_layout.operator(
            dependency_utils.InstallDependenciesOperator.bl_idname,
            text=f"Install {dependency.gui_name}",
            icon="CONSOLE",
        )
        install_dependency_props.dependency_package_name = package_name
        # Add operator to second column
        install_dependency_props = column_flow_layout.operator(
            dependency_utils.UninstallDependenciesOperator.bl_idname,
            text=f"Remove {dependency.gui_name}",
            icon="CONSOLE",
        )
        install_dependency_props.dependency_package_name = package_name

    def _draw_dependencies(self, install_dependency_box):
        install_dependency_box.label(text="Dependencies:")
        install_dependency_button_box = install_dependency_box.column_flow(columns=2)
        install_dependency_props = install_dependency_button_box.operator(
            dependency_utils.InstallDependenciesOperator.bl_idname, icon="CONSOLE")
        install_dependency_props.dependency_package_name = ""

        remove_dependency_button_box = install_dependency_box.column_flow(columns=2)
        uninstall_dependency_props = remove_dependency_button_box.operator(
            dependency_utils.UninstallDependenciesOperator.bl_idname, icon="CONSOLE")
        uninstall_dependency_props.dependency_package_name = ""

        row = install_dependency_box.row()
        row.label(text="Pip Installation Status:")
        setuptools_missing_str = "Install setuptools (and restart Blender) to show version and location"
        removed_in_current_sesssion_str = ("(Restart Blender to clear imported module)")

        pip_manager = dependency_utils.PipManager.get_singleton()
        pip_status_box = install_dependency_box.box()
        pip_status_box_split = pip_status_box.split()
        pip_status_box_1 = pip_status_box_split.column()
        pip_status_box_2 = pip_status_box_split.column()
        pip_status_box_3 = pip_status_box_split.column()
        pip_status_box_4 = pip_status_box_split.column()

        pip_installation_status = (pip_manager.pip_dependency_status.installation_status)
        pip_installation_status_str = self._get_installation_status_str(pip_installation_status)
        version_str, location_str = pip_manager.get_package_info()
        version_str = self._get_package_info_str(version_str, pip_installation_status, setuptools_missing_str)
        location_str = self._get_package_info_str(location_str, pip_installation_status)

        pip_status_box_1.label(text="Pip")
        pip_status_box_2.label(text=f"{pip_installation_status_str}")
        pip_status_box_3.label(text=f"{version_str}")
        pip_status_box_4.label(text=f"{location_str}")

        row = install_dependency_box.row()
        row.label(text="Dependency Installation Status:")
        dependencies_status_box = install_dependency_box.box()

        dependency_manager = dependency_utils.DependencyManager.get_singleton()
        dependencies = dependency_manager.get_dependencies()
        for dependency in dependencies:
            self._draw_dependency(
                dependency,
                setuptools_missing_str,
                removed_in_current_sesssion_str,
                dependencies_status_box,
            )

        dependency_status_description_box = dependencies_status_box.box()
        dependency_status_description_box.label(
            text="Note: After uninstalling the dependencies one must restart"
            " Blender to clear the references to the module within Blender.")

        row = install_dependency_box.row()
        row.label(text="Added system paths:")

        sys_paths_box = install_dependency_box.box()
        addon_name = "polychase"

        prefs = bpy.context.preferences.addons[addon_name].preferences
        try:
            if prefs.sys_path_list_str != "[]":
                for entry in json.loads(prefs.sys_path_list_str):
                    sys_paths_box.label(text=f"{entry}")
            else:
                sys_paths_box.label(text="None")
        except json.decoder.JSONDecodeError:
            sys_paths_box.label(text="Error: Could not parse paths!")

        sys_paths_description_box = sys_paths_box.box()
        dependency_description = (
            "Note: When importing the installed dependencies, potentially not"
            " all required modules may be found - since Blender modifies the"
            " path available in sys.path. To prevent this, the addon will add"
            " the missing system paths (i.e. the paths that have been"
            " available during installation of the dependencies) to Blender's"
            " sys.path.")
        add_multi_line_label(sys_paths_description_box, dependency_description)


def add_multi_line_label(ui_layout, long_text, max_line_length=120):
    text_wrapper = textwrap.TextWrapper(width=max_line_length)
    wrapped_lines = text_wrapper.wrap(text=long_text)
    row = ui_layout.row()
    for text in wrapped_lines:
        row = ui_layout.row()
        row.scale_y = 0.25
        row.label(text=text)
