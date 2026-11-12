extern "C" __global__ void __launch_bounds__(128) sc0_fwd(
    const float* __restrict__ m, const float* __restrict__ d, const float* __restrict__ w,
    const int* __restrict__ sp, float* __restrict__ y) {
  const int a = blockIdx.x, ch = threadIdx.x;
  const float* xr = m + (long long)a * 2048 + ch;
  const float* wr = w + (long long)sp[a] * 1664 + ch;
  const float s = 1.0f / (d[a] + 1.0f);
  const float x0 = __ldg(xr + 0) * s;
  const float x1 = __ldg(xr + 128) * s;
  const float x2 = __ldg(xr + 256) * s;
  const float x3 = __ldg(xr + 384) * s;
  const float x4 = __ldg(xr + 512) * s;
  const float x5 = __ldg(xr + 640) * s;
  const float x6 = __ldg(xr + 768) * s;
  const float x7 = __ldg(xr + 896) * s;
  const float x8 = __ldg(xr + 1024) * s;
  const float x9 = __ldg(xr + 1152) * s;
  const float x10 = __ldg(xr + 1280) * s;
  const float x11 = __ldg(xr + 1408) * s;
  const float x12 = __ldg(xr + 1536) * s;
  const float x13 = __ldg(xr + 1664) * s;
  const float x14 = __ldg(xr + 1792) * s;
  const float x15 = __ldg(xr + 1920) * s;
  const float v0 = x0 * x0;
  const float v1 = x0 * v0;
  const float v2 = x1 * x1;
  const float v3 = x2 * x2;
  const float v4 = x3 * x3;
  const float v5 = fmaf(0x1.279a740000000p-1f, v4, fmaf(0x1.279a740000000p-1f, v3, 0x1.279a740000000p-1f * v2));
  const float v6 = x0 * v5;
  const float v7 = x4 * x4;
  const float v8 = x5 * x5;
  const float v9 = x6 * x6;
  const float v10 = x7 * x7;
  const float v11 = x8 * x8;
  const float v12 = fmaf(0x1.c9f25c0000000p-2f, v11, fmaf(0x1.c9f25c0000000p-2f, v10, fmaf(0x1.c9f25c0000000p-2f, v9, fmaf(0x1.c9f25c0000000p-2f, v8, 0x1.c9f25c0000000p-2f * v7))));
  const float v13 = x0 * v12;
  const float v14 = x9 * x9;
  const float v15 = x10 * x10;
  const float v16 = x11 * x11;
  const float v17 = x12 * x12;
  const float v18 = x13 * x13;
  const float v19 = x14 * x14;
  const float v20 = x15 * x15;
  const float v21 = fmaf(0x1.83091e0000000p-2f, v20, fmaf(0x1.83091e0000000p-2f, v19, fmaf(0x1.83091e0000000p-2f, v18, fmaf(0x1.83091e0000000p-2f, v17, fmaf(0x1.83091e0000000p-2f, v16, fmaf(0x1.83091e0000000p-2f, v15, 0x1.83091e0000000p-2f * v14))))));
  const float v22 = x0 * v21;
  const float v23 = x3 * x1;
  const float v24 = 0x1.6a09e60000000p+0f * v23;
  const float v25 = x2 * x1;
  const float v26 = 0x1.6a09e60000000p+0f * v25;
  const float v27 = fmaf(-0x1.a20bd80000000p-2f, v4, fmaf(0x1.a20bd80000000p-1f, v3, -0x1.a20bd80000000p-2f * v2));
  const float v28 = x3 * x2;
  const float v29 = 0x1.6a09e60000000p+0f * v28;
  const float v30 = fmaf(0x1.6a09e60000000p-1f, v4, -0x1.6a09e60000000p-1f * v2);
  const float v31 = 0x1.c9f25c0000000p-2f * x4;
  const float v32 = 0x1.c9f25c0000000p-2f * x5;
  const float v33 = 0x1.c9f25c0000000p-2f * x6;
  const float v34 = 0x1.c9f25c0000000p-2f * x7;
  const float v35 = 0x1.c9f25c0000000p-2f * x8;
  const float v36 = fmaf(v30, v35, fmaf(v29, v34, fmaf(v27, v33, fmaf(v26, v32, v24 * v31))));
  const float v37 = x6 * x4;
  const float v38 = x7 * x5;
  const float v39 = fmaf(0x1.da05180000000p-1f, v38, -0x1.11acee0000000p+0f * v37);
  const float v40 = x7 * x4;
  const float v41 = x6 * x5;
  const float v42 = x8 * x5;
  const float v43 = fmaf(-0x1.da05180000000p-1f, v42, fmaf(0x1.11acee0000000p-1f, v41, 0x1.da05180000000p-1f * v40));
  const float v44 = fmaf(-0x1.11acee0000000p-1f, v11, fmaf(0x1.11acee0000000p-2f, v10, fmaf(0x1.11acee0000000p-1f, v9, fmaf(0x1.11acee0000000p-2f, v8, -0x1.11acee0000000p-1f * v7))));
  const float v45 = x5 * x4;
  const float v46 = x7 * x6;
  const float v47 = x8 * x7;
  const float v48 = fmaf(0x1.da05180000000p-1f, v47, fmaf(0x1.11acee0000000p-1f, v46, 0x1.da05180000000p-1f * v45));
  const float v49 = x8 * x6;
  const float v50 = fmaf(-0x1.11acee0000000p+0f, v49, fmaf(0x1.da05180000000p-2f, v10, -0x1.da05180000000p-2f * v8));
  const float v51 = fmaf(v35, v50, fmaf(v34, v48, fmaf(v33, v44, fmaf(v32, v43, v31 * v39))));
  const float v52 = x13 * x1;
  const float v53 = x15 * x1;
  const float v54 = x10 * x2;
  const float v55 = x9 * x3;
  const float v56 = x11 * x3;
  const float v57 = fmaf(-0x1.3c03660000000p-3f, v56, fmaf(0x1.31fa800000000p-1f, v55, fmaf(0x1.f3a92c0000000p-2f, v54, fmaf(-0x1.31fa800000000p-1f, v53, -0x1.3c03660000000p-3f * v52))));
  const float v58 = x12 * x1;
  const float v59 = x14 * x1;
  const float v60 = x11 * x2;
  const float v61 = x10 * x3;
  const float v62 = fmaf(0x1.f3a92c0000000p-2f, v61, fmaf(0x1.3c03660000000p-1f, v60, fmaf(-0x1.f3a92c0000000p-2f, v59, -0x1.83091e0000000p-2f * v58)));
  const float v63 = x11 * x1;
  const float v64 = x12 * x2;
  const float v65 = x13 * x3;
  const float v66 = fmaf(0x1.11acee0000000p-1f, v65, fmaf(0x1.4f2ec40000000p-1f, v64, 0x1.11acee0000000p-1f * v63));
  const float v67 = x10 * x1;
  const float v68 = x13 * x2;
  const float v69 = x12 * x3;
  const float v70 = x14 * x3;
  const float v71 = fmaf(0x1.f3a92c0000000p-2f, v70, fmaf(-0x1.83091e0000000p-2f, v69, fmaf(0x1.3c03660000000p-1f, v68, 0x1.f3a92c0000000p-2f * v67)));
  const float v72 = x9 * x1;
  const float v73 = x14 * x2;
  const float v74 = x15 * x3;
  const float v75 = fmaf(0x1.31fa800000000p-1f, v74, fmaf(0x1.f3a92c0000000p-2f, v73, fmaf(0x1.31fa800000000p-1f, v72, fmaf(-0x1.3c03660000000p-3f, v65, 0x1.3c03660000000p-3f * v63))));
  const float v76 = fmaf(v35, v75, fmaf(v34, v71, fmaf(v33, v66, fmaf(v32, v62, v31 * v57))));
  const float v77 = x13 * x9;
  const float v78 = x12 * x10;
  const float v79 = x13 * x11;
  const float v80 = x15 * x11;
  const float v81 = fmaf(0x1.f3a92c0000000p-2f, v80, fmaf(0x1.83091e0000000p-1f, v79, fmaf(-0x1.f3a92c0000000p-1f, v78, -0x1.f3a92c0000000p-2f * v77)));
  const float v82 = x14 * x9;
  const float v83 = x13 * x10;
  const float v84 = x15 * x10;
  const float v85 = x12 * x11;
  const float v86 = x14 * x11;
  const float v87 = fmaf(-0x1.31fa800000000p-1f, v86, fmaf(0x1.3c03660000000p-2f, v85, fmaf(-0x1.8b043e0000000p-1f, v84, fmaf(0x1.31fa800000000p-1f, v83, 0x1.8b043e0000000p-1f * v82))));
  const float v88 = fmaf(-0x1.1751a40000000p-1f, v20, fmaf(0x1.4f2ec40000000p-2f, v18, fmaf(0x1.bee9060000000p-2f, v17, fmaf(0x1.4f2ec40000000p-2f, v16, -0x1.1751a40000000p-1f * v14))));
  const float v89 = x10 * x9;
  const float v90 = x11 * x10;
  const float v91 = x13 * x12;
  const float v92 = x14 * x13;
  const float v93 = x15 * x14;
  const float v94 = fmaf(0x1.8b043e0000000p-1f, v93, fmaf(0x1.31fa800000000p-1f, v92, fmaf(0x1.3c03660000000p-2f, v91, fmaf(0x1.31fa800000000p-1f, v90, 0x1.8b043e0000000p-1f * v89))));
  const float v95 = x11 * x9;
  const float v96 = x14 * x12;
  const float v97 = x15 * x13;
  const float v98 = fmaf(-0x1.f3a92c0000000p-2f, v97, fmaf(-0x1.f3a92c0000000p-1f, v96, fmaf(-0x1.f3a92c0000000p-2f, v95, fmaf(0x1.83091e0000000p-2f, v18, -0x1.83091e0000000p-2f * v16))));
  const float v99 = fmaf(v35, v98, fmaf(v34, v94, fmaf(v33, v88, fmaf(v32, v87, v31 * v81))));
  float y0 = 0.0f;
  { const float wb = __ldg(wr + 0);
    y0 = fmaf(wb, x0, y0);
  }
  { const float wb = __ldg(wr + 128);
    y0 = fmaf(wb, v0, y0);
  }
  { const float wb = __ldg(wr + 256);
    y0 = fmaf(wb, v1, y0);
  }
  { const float wb = __ldg(wr + 384);
    y0 = fmaf(wb, v5, y0);
  }
  { const float wb = __ldg(wr + 512);
    y0 = fmaf(wb, v6, y0);
  }
  { const float wb = __ldg(wr + 640);
    y0 = fmaf(wb, v12, y0);
  }
  { const float wb = __ldg(wr + 768);
    y0 = fmaf(wb, v13, y0);
  }
  { const float wb = __ldg(wr + 896);
    y0 = fmaf(wb, v21, y0);
  }
  { const float wb = __ldg(wr + 1024);
    y0 = fmaf(wb, v22, y0);
  }
  { const float wb = __ldg(wr + 1152);
    y0 = fmaf(wb, v36, y0);
  }
  { const float wb = __ldg(wr + 1280);
    y0 = fmaf(wb, v51, y0);
  }
  { const float wb = __ldg(wr + 1408);
    y0 = fmaf(wb, v76, y0);
  }
  { const float wb = __ldg(wr + 1536);
    y0 = fmaf(wb, v99, y0);
  }
  y[(long long)a * 128 + 0 + ch] = y0;
}
