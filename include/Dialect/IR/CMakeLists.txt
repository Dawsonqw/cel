add_mlir_dialect(CelOps Cel)

set(LLVM_TARGET_DEFINITIONS CelOps.td)
mlir_tablegen(CelOps.h.inc -gen-op-decls)
mlir_tablegen(CelOps.cpp.inc -gen-op-defs)
mlir_tablegen(CelDialect.h.inc -gen-dialect-decls)
mlir_tablegen(CelDialect.cpp.inc -gen-dialect-defs)

mlir_tablegen(Cel.md -gen-dialect-doc)

add_public_tablegen_target(cel_dialect_gen)