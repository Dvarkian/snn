; ModuleID = '<string>'
source_filename = "kernel"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.RuntimeContext.7 = type { i8*, %struct.LLVMRuntime.6*, i32, i64* }
%struct.LLVMRuntime.6 = type { %struct.PreallocatedMemoryChunk.1, %struct.PreallocatedMemoryChunk.1, i8* (i8*, i64, i64)*, void (i8*)*, void (i8*, ...)*, i32 (i8*, i64, i8*, %struct.__va_list_tag.2*)*, i8*, [512 x i8*], [512 x i64], i8*, void (i8*, i32, i32, i8*, void (i8*, i32, i32)*)*, [1024 x %struct.ListManager.3*], [1024 x %struct.NodeManager.4*], [1024 x i8*], i8*, %struct.RandState.5*, i8*, void (i8*, i8*)*, void (i8*)*, [2048 x i8], [32 x i64], i32, i64, i8*, i32, i32, i64 }
%struct.PreallocatedMemoryChunk.1 = type { i8*, i8*, i64 }
%struct.__va_list_tag.2 = type { i32, i32, i8*, i8* }
%struct.ListManager.3 = type { [131072 x i8*], i64, i64, i32, i32, i32, %struct.LLVMRuntime.6* }
%struct.NodeManager.4 = type <{ %struct.LLVMRuntime.6*, i32, i32, i32, i32, %struct.ListManager.3*, %struct.ListManager.3*, %struct.ListManager.3*, i32, [4 x i8] }>
%struct.RandState.5 = type { i32, i32, i32, i32, i32 }

; Function Attrs: nofree norecurse nosync nounwind
define void @_sparse_csr_matvec_transpose_heter_cpu_c78_0_kernel_0_serial(%struct.RuntimeContext.7* nocapture readonly %context) local_unnamed_addr #0 {
entry:
  %0 = bitcast %struct.RuntimeContext.7* %context to { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }**
  %1 = load { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }*, { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }** %0, align 8
  %2 = getelementptr { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }, { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }* %1, i64 0, i32 2, i32 0, i32 0
  %3 = load i32, i32* %2, align 4
  %4 = add i32 %3, -1
  %5 = icmp sgt i32 %4, 0
  br i1 %5, label %for_loop_body.preheader, label %after_for

for_loop_body.preheader:                          ; preds = %entry
  %wide.trip.count13 = zext i32 %4 to i64
  br label %for_loop_body

for_loop_body:                                    ; preds = %for_loop_test.loopexit, %for_loop_body.preheader
  %indvars.iv10 = phi i64 [ 0, %for_loop_body.preheader ], [ %indvars.iv.next11, %for_loop_test.loopexit ]
  %6 = load { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }*, { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }** %0, align 8
  %7 = getelementptr { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }, { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }* %6, i64 0, i32 2, i32 1
  %8 = load i32*, i32** %7, align 8
  %9 = getelementptr i32, i32* %8, i64 %indvars.iv10
  %10 = load i32, i32* %9, align 4
  %indvars.iv.next11 = add nuw nsw i64 %indvars.iv10, 1
  %11 = getelementptr i32, i32* %9, i64 1
  %12 = load i32, i32* %11, align 4
  %13 = getelementptr { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }, { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }* %6, i64 0, i32 3, i32 1
  %14 = load float*, float** %13, align 8
  %15 = getelementptr float, float* %14, i64 %indvars.iv10
  %16 = load float, float* %15, align 4
  %17 = icmp slt i32 %10, %12
  br i1 %17, label %for_loop_body1.lr.ph, label %for_loop_test.loopexit

for_loop_body1.lr.ph:                             ; preds = %for_loop_body
  %18 = getelementptr { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }, { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }* %6, i64 0, i32 0, i32 1
  %19 = getelementptr { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }, { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }* %6, i64 0, i32 1, i32 1
  %20 = getelementptr { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }, { { { i32 }, float* }, { { i32 }, i32* }, { { i32 }, i32* }, { { i32 }, float* }, { { i32 }, float* } }* %6, i64 0, i32 4, i32 1
  %21 = sext i32 %10 to i64
  %wide.trip.count = sext i32 %12 to i64
  %22 = sub nsw i64 %wide.trip.count, %21
  %23 = xor i64 %21, -1
  %24 = add nsw i64 %23, %wide.trip.count
  %xtraiter = and i64 %22, 3
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %for_loop_body1.prol.loopexit, label %for_loop_body1.prol.preheader

for_loop_body1.prol.preheader:                    ; preds = %for_loop_body1.lr.ph
  br label %for_loop_body1.prol

for_loop_body1.prol:                              ; preds = %for_loop_body1.prol, %for_loop_body1.prol.preheader
  %lsr.iv = phi i64 [ %xtraiter, %for_loop_body1.prol.preheader ], [ %lsr.iv.next, %for_loop_body1.prol ]
  %indvars.iv.prol = phi i64 [ %indvars.iv.next.prol, %for_loop_body1.prol ], [ %21, %for_loop_body1.prol.preheader ]
  %25 = load float*, float** %18, align 8
  %scevgep15 = getelementptr float, float* %25, i64 %indvars.iv.prol
  %26 = load float, float* %scevgep15, align 4
  %27 = fmul reassoc ninf nsz float %26, %16
  %28 = load i32*, i32** %19, align 8
  %scevgep = getelementptr i32, i32* %28, i64 %indvars.iv.prol
  %29 = load i32, i32* %scevgep, align 4
  %30 = load float*, float** %20, align 8
  %31 = sext i32 %29 to i64
  %32 = getelementptr float, float* %30, i64 %31
  %33 = load float, float* %32, align 4
  %34 = fadd reassoc ninf nsz float %33, %27
  store float %34, float* %32, align 4
  %indvars.iv.next.prol = add nsw i64 %indvars.iv.prol, 1
  %lsr.iv.next = add nsw i64 %lsr.iv, -1
  %prol.iter.cmp.not = icmp eq i64 %lsr.iv.next, 0
  br i1 %prol.iter.cmp.not, label %for_loop_body1.prol.loopexit.loopexit, label %for_loop_body1.prol, !llvm.loop !6

for_loop_body1.prol.loopexit.loopexit:            ; preds = %for_loop_body1.prol
  br label %for_loop_body1.prol.loopexit

for_loop_body1.prol.loopexit:                     ; preds = %for_loop_body1.prol.loopexit.loopexit, %for_loop_body1.lr.ph
  %indvars.iv.unr = phi i64 [ %21, %for_loop_body1.lr.ph ], [ %indvars.iv.next.prol, %for_loop_body1.prol.loopexit.loopexit ]
  %35 = icmp ult i64 %24, 3
  br i1 %35, label %for_loop_test.loopexit, label %for_loop_body1.preheader

for_loop_body1.preheader:                         ; preds = %for_loop_body1.prol.loopexit
  br label %for_loop_body1

after_for.loopexit:                               ; preds = %for_loop_test.loopexit
  br label %after_for

after_for:                                        ; preds = %after_for.loopexit, %entry
  ret void

for_loop_test.loopexit.loopexit:                  ; preds = %for_loop_body1
  br label %for_loop_test.loopexit

for_loop_test.loopexit:                           ; preds = %for_loop_test.loopexit.loopexit, %for_loop_body1.prol.loopexit, %for_loop_body
  %exitcond14.not = icmp eq i64 %indvars.iv.next11, %wide.trip.count13
  br i1 %exitcond14.not, label %after_for.loopexit, label %for_loop_body

for_loop_body1:                                   ; preds = %for_loop_body1, %for_loop_body1.preheader
  %indvars.iv = phi i64 [ %indvars.iv.next.3, %for_loop_body1 ], [ %indvars.iv.unr, %for_loop_body1.preheader ]
  %36 = load float*, float** %18, align 8
  %scevgep24 = getelementptr float, float* %36, i64 %indvars.iv
  %37 = load float, float* %scevgep24, align 4
  %38 = fmul reassoc ninf nsz float %37, %16
  %39 = load i32*, i32** %19, align 8
  %scevgep25 = getelementptr i32, i32* %39, i64 %indvars.iv
  %40 = load i32, i32* %scevgep25, align 4
  %41 = load float*, float** %20, align 8
  %42 = sext i32 %40 to i64
  %43 = getelementptr float, float* %41, i64 %42
  %44 = load float, float* %43, align 4
  %45 = fadd reassoc ninf nsz float %44, %38
  store float %45, float* %43, align 4
  %46 = load float*, float** %18, align 8
  %scevgep28 = getelementptr float, float* %46, i64 1
  %scevgep29 = getelementptr float, float* %scevgep28, i64 %indvars.iv
  %47 = load float, float* %scevgep29, align 4
  %48 = fmul reassoc ninf nsz float %47, %16
  %49 = load i32*, i32** %19, align 8
  %scevgep26 = getelementptr i32, i32* %49, i64 1
  %scevgep27 = getelementptr i32, i32* %scevgep26, i64 %indvars.iv
  %50 = load i32, i32* %scevgep27, align 4
  %51 = load float*, float** %20, align 8
  %52 = sext i32 %50 to i64
  %53 = getelementptr float, float* %51, i64 %52
  %54 = load float, float* %53, align 4
  %55 = fadd reassoc ninf nsz float %54, %48
  store float %55, float* %53, align 4
  %56 = load float*, float** %18, align 8
  %scevgep22 = getelementptr float, float* %56, i64 2
  %scevgep23 = getelementptr float, float* %scevgep22, i64 %indvars.iv
  %57 = load float, float* %scevgep23, align 4
  %58 = fmul reassoc ninf nsz float %57, %16
  %59 = load i32*, i32** %19, align 8
  %scevgep20 = getelementptr i32, i32* %59, i64 2
  %scevgep21 = getelementptr i32, i32* %scevgep20, i64 %indvars.iv
  %60 = load i32, i32* %scevgep21, align 4
  %61 = load float*, float** %20, align 8
  %62 = sext i32 %60 to i64
  %63 = getelementptr float, float* %61, i64 %62
  %64 = load float, float* %63, align 4
  %65 = fadd reassoc ninf nsz float %64, %58
  store float %65, float* %63, align 4
  %66 = load float*, float** %18, align 8
  %scevgep18 = getelementptr float, float* %66, i64 3
  %scevgep19 = getelementptr float, float* %scevgep18, i64 %indvars.iv
  %67 = load float, float* %scevgep19, align 4
  %68 = fmul reassoc ninf nsz float %67, %16
  %69 = load i32*, i32** %19, align 8
  %scevgep16 = getelementptr i32, i32* %69, i64 3
  %scevgep17 = getelementptr i32, i32* %scevgep16, i64 %indvars.iv
  %70 = load i32, i32* %scevgep17, align 4
  %71 = load float*, float** %20, align 8
  %72 = sext i32 %70 to i64
  %73 = getelementptr float, float* %71, i64 %72
  %74 = load float, float* %73, align 4
  %75 = fadd reassoc ninf nsz float %74, %68
  store float %75, float* %73, align 4
  %indvars.iv.next.3 = add nsw i64 %indvars.iv, 4
  %exitcond.not.3 = icmp eq i64 %wide.trip.count, %indvars.iv.next.3
  br i1 %exitcond.not.3, label %for_loop_test.loopexit.loopexit, label %for_loop_body1
}

attributes #0 = { nofree norecurse nosync nounwind }

!llvm.ident = !{!0}
!llvm.module.flags = !{!1, !2, !3, !4, !5}

!0 = !{!"Ubuntu clang version 14.0.6"}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 1}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.unroll.disable"}
