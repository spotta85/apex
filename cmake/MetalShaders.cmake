# add_metal_library(<target> OUTPUT <name.metallib> SOURCES a.metal ... [DEPENDS h1 h2 ...])
#
# By default the shaders are compiled with `xcrun -sdk macosx metal`. If Xcode's Metal
# toolchain component is out of date (xcrun reports "missing Metal Toolchain" after an Xcode
# update), point APEX_METAL_BIN_DIR at a directory holding `metal` and `metallib`, e.g. the
# still-mounted previous toolchain under /private/var/run/com.apple.security.cryptexd/mnt/.
# The proper fix is `xcodebuild -downloadComponent MetalToolchain`.
set(APEX_METAL_BIN_DIR "" CACHE PATH "Directory with metal/metallib (empty: use xcrun)")
if(APEX_METAL_BIN_DIR)
  set(APEX_METAL_CC ${APEX_METAL_BIN_DIR}/metal)
  set(APEX_METAL_LIB ${APEX_METAL_BIN_DIR}/metallib)
else()
  set(APEX_METAL_CC xcrun -sdk macosx metal)
  set(APEX_METAL_LIB xcrun -sdk macosx metallib)
endif()

function(add_metal_library target)
  cmake_parse_arguments(ARG "" "OUTPUT" "SOURCES;DEPENDS" ${ARGN})
  set(deps)
  foreach(dep ${ARG_DEPENDS})
    list(APPEND deps ${CMAKE_CURRENT_SOURCE_DIR}/${dep})
  endforeach()
  set(airs)
  foreach(src ${ARG_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    set(air ${CMAKE_CURRENT_BINARY_DIR}/${name}.air)
    add_custom_command(OUTPUT ${air}
      COMMAND ${APEX_METAL_CC} -O2 -fno-fast-math -c ${CMAKE_CURRENT_SOURCE_DIR}/${src} -o ${air}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src} ${deps}
      COMMENT "metal ${src}")
    list(APPEND airs ${air})
  endforeach()
  set(lib ${CMAKE_CURRENT_BINARY_DIR}/${ARG_OUTPUT})
  add_custom_command(OUTPUT ${lib}
    COMMAND ${APEX_METAL_LIB} ${airs} -o ${lib}
    DEPENDS ${airs}
    COMMENT "metallib ${ARG_OUTPUT}")
  add_custom_target(${target} DEPENDS ${lib})
endfunction()
