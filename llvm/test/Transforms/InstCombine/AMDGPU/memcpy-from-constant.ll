; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -amdgpu-aa-wrapper -amdgpu-aa -instcombine -o - %s | FileCheck %s

; Make sure the optimization from memcpy-from-global.ll happens, but
; the constant source is not a global variable.

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; Simple memcpy to alloca from constant address space argument.
define i8 @memcpy_constant_arg_ptr_to_alloca([32 x i8] addrspace(4)* noalias readonly align 4 dereferenceable(32) %arg, i32 %idx) {
; CHECK-LABEL: @memcpy_constant_arg_ptr_to_alloca(
; CHECK-NEXT:    [[TMP1:%.*]] = sext i32 [[IDX:%.*]] to i64
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr [32 x i8], [32 x i8] addrspace(4)* [[ARG:%.*]], i64 0, i64 [[TMP1]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i8, i8 addrspace(4)* [[GEP]], align 1
; CHECK-NEXT:    ret i8 [[LOAD]]
;
  %alloca = alloca [32 x i8], align 4, addrspace(5)
  %alloca.cast = bitcast [32 x i8] addrspace(5)* %alloca to i8 addrspace(5)*
  %arg.cast = bitcast [32 x i8] addrspace(4)* %arg to i8 addrspace(4)*
  call void @llvm.memcpy.p5i8.p4i8.i64(i8 addrspace(5)* %alloca.cast, i8 addrspace(4)* %arg.cast, i64 32, i1 false)
  %gep = getelementptr inbounds [32 x i8], [32 x i8] addrspace(5)* %alloca, i32 0, i32 %idx
  %load = load i8, i8 addrspace(5)* %gep
  ret i8 %load
}

; Simple memcpy to alloca from constant address space intrinsic call
define amdgpu_kernel void @memcpy_constant_intrinsic_ptr_to_alloca(i8 addrspace(1)* %out, i32 %idx) {
; CHECK-LABEL: @memcpy_constant_intrinsic_ptr_to_alloca(
; CHECK-NEXT:    [[KERNARG_SEGMENT_PTR:%.*]] = call align 16 dereferenceable(32) i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
; CHECK-NEXT:    [[TMP1:%.*]] = sext i32 [[IDX:%.*]] to i64
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr i8, i8 addrspace(4)* [[KERNARG_SEGMENT_PTR]], i64 [[TMP1]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i8, i8 addrspace(4)* [[GEP]], align 1
; CHECK-NEXT:    store i8 [[LOAD]], i8 addrspace(1)* [[OUT:%.*]], align 1
; CHECK-NEXT:    ret void
;
  %alloca = alloca [32 x i8], align 4, addrspace(5)
  %alloca.cast = bitcast [32 x i8] addrspace(5)* %alloca to i8 addrspace(5)*
  %kernarg.segment.ptr = call dereferenceable(32) align 16 i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  call void @llvm.memcpy.p5i8.p4i8.i64(i8 addrspace(5)* %alloca.cast, i8 addrspace(4)* %kernarg.segment.ptr, i64 32, i1 false)
  %gep = getelementptr inbounds [32 x i8], [32 x i8] addrspace(5)* %alloca, i32 0, i32 %idx
  %load = load i8, i8 addrspace(5)* %gep
  store i8 %load, i8 addrspace(1)* %out
  ret void
}

; Alloca is written through a flat pointer
define i8 @memcpy_constant_arg_ptr_to_alloca_addrspacecast_to_flat([32 x i8] addrspace(4)* noalias readonly align 4 dereferenceable(32) %arg, i32 %idx) {
; CHECK-LABEL: @memcpy_constant_arg_ptr_to_alloca_addrspacecast_to_flat(
; CHECK-NEXT:    [[TMP1:%.*]] = sext i32 [[IDX:%.*]] to i64
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr [32 x i8], [32 x i8] addrspace(4)* [[ARG:%.*]], i64 0, i64 [[TMP1]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i8, i8 addrspace(4)* [[GEP]], align 1
; CHECK-NEXT:    ret i8 [[LOAD]]
;
  %alloca = alloca [32 x i8], align 4, addrspace(5)
  %alloca.cast = bitcast [32 x i8] addrspace(5)* %alloca to i8 addrspace(5)*
  %alloca.cast.asc = addrspacecast i8 addrspace(5)* %alloca.cast to i8*
  %arg.cast = bitcast [32 x i8] addrspace(4)* %arg to i8 addrspace(4)*
  call void @llvm.memcpy.p0i8.p4i8.i64(i8* %alloca.cast.asc, i8 addrspace(4)* %arg.cast, i64 32, i1 false)
  %gep = getelementptr inbounds [32 x i8], [32 x i8] addrspace(5)* %alloca, i32 0, i32 %idx
  %load = load i8, i8 addrspace(5)* %gep
  ret i8 %load
}

; Alloca is only addressed through flat pointer.
define i8 @memcpy_constant_arg_ptr_to_alloca_addrspacecast_to_flat2([32 x i8] addrspace(4)* noalias readonly align 4 dereferenceable(32) %arg, i32 %idx) {
; CHECK-LABEL: @memcpy_constant_arg_ptr_to_alloca_addrspacecast_to_flat2(
; CHECK-NEXT:    [[ALLOCA:%.*]] = alloca [32 x i8], align 4, addrspace(5)
; CHECK-NEXT:    [[ALLOCA_CAST1:%.*]] = getelementptr inbounds [32 x i8], [32 x i8] addrspace(5)* [[ALLOCA]], i32 0, i32 0
; CHECK-NEXT:    [[ALLOCA_CAST:%.*]] = addrspacecast i8 addrspace(5)* [[ALLOCA_CAST1]] to i8*
; CHECK-NEXT:    [[ARG_CAST:%.*]] = getelementptr inbounds [32 x i8], [32 x i8] addrspace(4)* [[ARG:%.*]], i64 0, i64 0
; CHECK-NEXT:    call void @llvm.memcpy.p0i8.p4i8.i64(i8* nonnull align 1 dereferenceable(32) [[ALLOCA_CAST]], i8 addrspace(4)* align 4 dereferenceable(32) [[ARG_CAST]], i64 32, i1 false)
; CHECK-NEXT:    [[GEP2:%.*]] = getelementptr inbounds [32 x i8], [32 x i8] addrspace(5)* [[ALLOCA]], i32 0, i32 [[IDX:%.*]]
; CHECK-NEXT:    [[GEP:%.*]] = addrspacecast i8 addrspace(5)* [[GEP2]] to i8*
; CHECK-NEXT:    [[LOAD:%.*]] = load i8, i8* [[GEP]], align 1
; CHECK-NEXT:    ret i8 [[LOAD]]
;
  %alloca = alloca [32 x i8], align 4, addrspace(5)
  %alloca.cast.asc = addrspacecast [32 x i8] addrspace(5)* %alloca to [32 x i8]*
  %alloca.cast = bitcast [32 x i8]* %alloca.cast.asc to i8*
  %arg.cast = bitcast [32 x i8] addrspace(4)* %arg to i8 addrspace(4)*
  call void @llvm.memcpy.p0i8.p4i8.i64(i8* %alloca.cast, i8 addrspace(4)* %arg.cast, i64 32, i1 false)
  %gep = getelementptr inbounds [32 x i8], [32 x i8]* %alloca.cast.asc, i32 0, i32 %idx
  %load = load i8, i8* %gep
  ret i8 %load
}

declare void @llvm.memcpy.p5i8.p4i8.i64(i8 addrspace(5)* nocapture, i8 addrspace(4)* nocapture, i64, i1) #0
declare void @llvm.memcpy.p0i8.p4i8.i64(i8* nocapture, i8 addrspace(4)* nocapture, i64, i1) #0
declare i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr() #1

attributes #0 = { argmemonly nounwind willreturn }
attributes #1 = { nounwind readnone speculatable }