# add_metal_library(<target> OUTPUT <name.metallib> SOURCES a.metal b.metal ...)
function(add_metal_library target)
  cmake_parse_arguments(ARG "" "OUTPUT" "SOURCES" ${ARGN})
  set(airs)
  foreach(src ${ARG_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    set(air ${CMAKE_CURRENT_BINARY_DIR}/${name}.air)
    add_custom_command(OUTPUT ${air}
      COMMAND xcrun -sdk macosx metal -O2 -c ${CMAKE_CURRENT_SOURCE_DIR}/${src} -o ${air}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
      COMMENT "metal ${src}")
    list(APPEND airs ${air})
  endforeach()
  set(lib ${CMAKE_CURRENT_BINARY_DIR}/${ARG_OUTPUT})
  add_custom_command(OUTPUT ${lib}
    COMMAND xcrun -sdk macosx metallib ${airs} -o ${lib}
    DEPENDS ${airs}
    COMMENT "metallib ${ARG_OUTPUT}")
  add_custom_target(${target} DEPENDS ${lib})
endfunction()
