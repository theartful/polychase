# Adapted from Synfig: https://github.com/synfig/synfig/blob/master/cmake/SynfigVcpkg.cmake
# which I actually wrote in a GSoC project.

get_filename_component(VCPKG_TOOLCHAIN_DIR "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
find_package (Python COMPONENTS Interpreter REQUIRED)

if(CMAKE_VERSION VERSION_LESS "3.14")
	message(WARNING "At least CMake 3.14 is required for installing the dependencies (current version: ${CMAKE_VERSION})")
endif()

function(install_app_dependencies)
	if(CMAKE_VERSION VERSION_LESS "3.14")
		return()
	endif()
	cmake_policy(SET CMP0087 NEW)

	cmake_parse_arguments(PARSE_ARGV 0 arg
		""
		"DESTINATION"
		"TARGETS;FILES"
	)

	if(DEFINED arg_UNPARSED_ARGUMENTS)
			message(FATAL_ERROR "${CMAKE_CURRENT_FUNCTION} was passed extra arguments: ${arg_UNPARSED_ARGUMENTS}")
	endif()
	if(NOT DEFINED arg_DESTINATION)
			message(FATAL_ERROR "DESTINATION must be specified")
	endif()

	if(NOT IS_ABSOLUTE "${arg_DESTINATION}")
			set(arg_DESTINATION "\${CMAKE_INSTALL_PREFIX}/${arg_DESTINATION}")
	endif()

	foreach(target IN LISTS arg_TARGETS arg_FILES)
		if(TARGET ${target})
			get_target_property(target_type "${target}" TYPE)
			if(target_type STREQUAL "INTERFACE_LIBRARY")
				continue()
			endif()
			set(name "${target}")
			set(file_path "$<TARGET_FILE:${target}>")
		else()
			get_filename_component(name "${target}" NAME_WE)
			set(file_path "${target}")
		endif()

		if(WIN32)
			install(CODE "
				message(\"-- Installing app dependencies for ${name}...\")
				execute_process(
					COMMAND \"${CMAKE_COMMAND}\" -E copy \"${file_path}\" \"${arg_DESTINATION}/${name}.tmp\"
				)
				execute_process(
					COMMAND powershell -noprofile -executionpolicy Bypass -file \"${VCPKG_TOOLCHAIN_DIR}/msbuild/applocal.ps1\"
						-targetBinary \"${arg_DESTINATION}/${name}.tmp\"
						-installedDir \"${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}$<$<CONFIG:Debug>:/debug>/bin\"
				)
				execute_process(
					COMMAND \"${CMAKE_COMMAND}\" -E remove \"${arg_DESTINATION}/${name}.tmp\"
				)"
			)
		elseif(UNIX)
			# TODO: make appdeps.py work on windows as well, and don't depend on vcpkg's
			# applocal.ps1, since it's not certain that it will remain the same
			set(APPDEPS "${CMAKE_SOURCE_DIR}/cmake/appdeps.py")
			install(CODE "
				message(\"-- Installing app dependencies for ${name}...\")
				execute_process(
					COMMAND \"${Python_EXECUTABLE}\" \"${APPDEPS}\" -L \"${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/$<$<CONFIG:Debug>:/debug>/lib\"
					--outdir \"${arg_DESTINATION}\" \"${file_path}\"
				)"
			)
		endif()
	endforeach()
endfunction()

# this closely follows what vcpkg does:
# https://github.com/microsoft/vcpkg/blob/01b29f6d8212bc845da64773b18665d682f5ab66/scripts/buildsystems/vcpkg.cmake#L697-L741
function(install)
	if(ARGV0 STREQUAL "TARGETS")
		# Will contain the list of targets
		set(parsed_targets "")
		set(component_param "")

		# Parse arguments given to the install function to find targets and (runtime) destination
		set(modifier "") # Modifier for the command in the argument
		set(last_command "") # Last command we found to process
		set(destination "")
		set(library_destination "")
		set(runtime_destination "")

		foreach(arg ${ARGV})
			if(arg MATCHES "^(ARCHIVE|LIBRARY|RUNTIME|OBJECTS|FRAMEWORK|BUNDLE|PRIVATE_HEADER|PUBLIC_HEADER|RESOURCE|INCLUDES)$")
				set(modifier "${arg}")
				continue()
			endif()
			if(arg MATCHES "^(TARGETS|DESTINATION|PERMISSIONS|CONFIGURATIONS|COMPONENT|NAMELINK_COMPONENT|OPTIONAL|EXCLUDE_FROM_ALL|NAMELINK_ONLY|NAMELINK_SKIP|EXPORT)$")
				set(last_command "${arg}")
				continue()
			endif()

			if(last_command STREQUAL "TARGETS")
				list(APPEND parsed_targets "${arg}")
			endif()

			if(last_command STREQUAL "DESTINATION" AND modifier STREQUAL "")
				set(destination "${arg}")
			endif()

			if(last_command STREQUAL "DESTINATION" AND modifier STREQUAL "RUNTIME")
				set(runtime_destination "${arg}")
			endif()

			if(last_command STREQUAL "DESTINATION" AND modifier STREQUAL "LIBRARY")
				set(library_destination "${arg}")
			endif()

			if(last_command STREQUAL "COMPONENT")
				set(component_param "COMPONENT" "${arg}")
			endif()
		endforeach()

		if(NOT destination)
			if(WIN32 AND runtime_destination)
				set(destination "${runtime_destination}")
			elseif(UNIX AND library_destination)
				set(destination "${library_destination}")
			endif()
		endif()

		# unlike vcpkg's implementation, which installs dependencies in the same
		# directory as the target, this will always installs dependencies inside
		# bin/lib directories
		if(WIN32)
			install_app_dependencies(TARGETS ${parsed_targets} DESTINATION ${destination})
		elseif(UNIX)
            set_target_properties(${parsed_targets} PROPERTIES INSTALL_RPATH "$ORIGIN")
			install_app_dependencies(TARGETS ${parsed_targets} DESTINATION ${destination})
		endif()
	endif()
	_install(${ARGV})
endfunction()
