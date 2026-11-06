extern "C" __global__ void cart_convert(const float* __restrict__ in0, const float* __restrict__ in1, const float* __restrict__ in2, const float* __restrict__ in3, float* __restrict__ out0, float* __restrict__ out1, float* __restrict__ out2, float* __restrict__ out3, long long nodes) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nodes * 1280) return;
    const long long node = t / 1280;
    const int rem = int(t - node * 1280);
    const int path = rem / 128, channel = rem - path * 128;
    switch (path) {
    case 0: {
        const float g0 = in0[node * 256 + 0 + channel];
        out0[node * 256 + 0 + channel] = 1.0f * g0;
        break; }
    case 1: {
        const float g0 = in0[node * 256 + 128 + channel];
        out0[node * 256 + 128 + channel] = 1.0f * g0;
        break; }
    case 2: {
        const float g0 = in1[node * 1152 + 0 + channel];
        const float g1 = in1[node * 1152 + 384 + channel];
        const float g2 = in1[node * 1152 + 768 + channel];
        out1[node * 1152 + 0 + channel] = 1.73205081f * g0 + -1.40056912e-16f * g1 + -9.07882191e-17f * g2;
        out1[node * 1152 + 384 + channel] = 5.17567294e-17f * g0 + 1.73205081f * g1 + 2.2019401e-17f * g2;
        out1[node * 1152 + 768 + channel] = -4.76174711e-17f * g0 + 7.09535666e-17f * g1 + 1.73205081f * g2;
        break; }
    case 3: {
        const float g0 = in1[node * 1152 + 128 + channel];
        const float g1 = in1[node * 1152 + 512 + channel];
        const float g2 = in1[node * 1152 + 896 + channel];
        out1[node * 1152 + 128 + channel] = 1.0f * g0 + 2.23248219e-17f * g1 + -1.39713215e-17f * g2;
        out1[node * 1152 + 512 + channel] = 5.76625166e-18f * g0 + 1.0f * g1 + -1.57755318e-17f * g2;
        out1[node * 1152 + 896 + channel] = -9.85150913e-19f * g0 + 7.81604749e-17f * g1 + 1.0f * g2;
        break; }
    case 4: {
        const float g0 = in1[node * 1152 + 256 + channel];
        const float g1 = in1[node * 1152 + 640 + channel];
        const float g2 = in1[node * 1152 + 1024 + channel];
        out1[node * 1152 + 256 + channel] = 2.12132034f * g0 + -1.00119825e-16f * g1 + -1.83312933e-16f * g2;
        out1[node * 1152 + 640 + channel] = -5.31590663e-17f * g0 + 2.12132034f * g1 + 1.04026668e-16f * g2;
        out1[node * 1152 + 1024 + channel] = 4.80631005e-17f * g0 + -1.18789192e-16f * g1 + 2.12132034f * g2;
        break; }
    case 5: {
        const float g0 = in2[node * 1920 + 0 + channel];
        const float g1 = in2[node * 1920 + 384 + channel];
        const float g2 = in2[node * 1920 + 768 + channel];
        const float g3 = in2[node * 1920 + 1152 + channel];
        const float g4 = in2[node * 1920 + 1536 + channel];
        out2[node * 1920 + 0 + channel] = 2.08909818e-16f * g0 + 1.31570223e-16f * g1 + -1.11803399f * g2 + -1.71075576e-16f * g3 + -1.93649167f * g4;
        out2[node * 1920 + 384 + channel] = 5.63940654e-17f * g0 + 1.93649167f * g1 + -4.55418429e-17f * g2 + 3.11302799e-16f * g3 + 1.82167372e-17f * g4;
        out2[node * 1920 + 768 + channel] = 1.93649167f * g0 + 6.13748943e-17f * g1 + 2.89700706e-16f * g2 + -1.31799015e-17f * g3 + -1.0189097e-16f * g4;
        out2[node * 1920 + 1152 + channel] = -6.04589843e-17f * g0 + 2.7353662e-16f * g1 + 2.23606798f * g2 + -1.70917357e-16f * g3 + -5.3119267e-18f * g4;
        out2[node * 1920 + 1536 + channel] = 3.88789374e-17f * g0 + 4.02354098e-16f * g1 + -5.09901722e-17f * g2 + 1.93649167f * g3 + -3.06848251e-16f * g4;
        break; }
    case 6: {
        const float g0 = in2[node * 1920 + 128 + channel];
        const float g1 = in2[node * 1920 + 512 + channel];
        const float g2 = in2[node * 1920 + 896 + channel];
        const float g3 = in2[node * 1920 + 1280 + channel];
        const float g4 = in2[node * 1920 + 1664 + channel];
        out2[node * 1920 + 128 + channel] = -7.32844211e-18f * g0 + 4.78303856e-18f * g1 + -0.353553391f * g2 + 3.27686889e-17f * g3 + -0.612372436f * g4;
        out2[node * 1920 + 512 + channel] = -2.32576117e-17f * g0 + 0.612372436f * g1 + -2.10067775e-17f * g2 + 4.61043766e-18f * g3 + -7.70958698e-18f * g4;
        out2[node * 1920 + 896 + channel] = 0.612372436f * g0 + 3.81981216e-17f * g1 + -7.69347027e-18f * g2 + 4.24252737e-18f * g3 + 1.17635986e-17f * g4;
        out2[node * 1920 + 1280 + channel] = 1.53316819e-17f * g0 + -7.93966816e-18f * g1 + 0.707106781f * g2 + -2.71423708e-17f * g3 + 1.31264545e-16f * g4;
        out2[node * 1920 + 1664 + channel] = -3.92922412e-18f * g0 + -1.02724308e-17f * g1 + 4.49956817e-17f * g2 + 0.612372436f * g3 + -1.79982727e-17f * g4;
        break; }
    case 7: {
        const float g0 = in2[node * 1920 + 256 + channel];
        const float g1 = in2[node * 1920 + 640 + channel];
        const float g2 = in2[node * 1920 + 1024 + channel];
        const float g3 = in2[node * 1920 + 1408 + channel];
        const float g4 = in2[node * 1920 + 1792 + channel];
        out2[node * 1920 + 256 + channel] = -5.69134388e-16f * g0 + 8.51975425e-17f * g1 + -1.44337567f * g2 + -5.06555173e-17f * g3 + -2.5f * g4;
        out2[node * 1920 + 640 + channel] = 2.36663609e-16f * g0 + 2.5f * g1 + -1.40268279e-16f * g3 + -4.19217619e-16f * g4;
        out2[node * 1920 + 1024 + channel] = 2.5f * g0 + 3.42706182e-16f * g1 + -1.91094783e-16f * g2 + -4.18805032e-16f * g3 + -1.86641226e-16f * g4;
        out2[node * 1920 + 1408 + channel] = -4.36684286e-16f * g0 + 1.29719676e-16f * g1 + 2.88675135f * g2 + 1.71368129e-16f * g3 + 4.63849289e-16f * g4;
        out2[node * 1920 + 1792 + channel] = 1.41483791e-17f * g0 + -5.8426566e-16f * g1 + 2.38957282e-16f * g2 + 2.5f * g3 + 3.72798122e-16f * g4;
        break; }
    case 8: {
        const float g0 = in3[node * 1792 + 0 + channel];
        const float g1 = in3[node * 1792 + 256 + channel];
        const float g2 = in3[node * 1792 + 512 + channel];
        const float g3 = in3[node * 1792 + 768 + channel];
        const float g4 = in3[node * 1792 + 1024 + channel];
        const float g5 = in3[node * 1792 + 1280 + channel];
        const float g6 = in3[node * 1792 + 1536 + channel];
        out3[node * 1792 + 0 + channel] = -2.09165007f * g0 + 1.86002847e-16f * g1 + -1.62018517f * g2 + 5.56966564e-16f * g3 + -2.60945405e-16f * g4 + 7.38861889e-16f * g5 + 3.33774157e-17f * g6;
        out3[node * 1792 + 256 + channel] = -1.45648946e-16f * g0 + 6.56726472e-17f * g1 + -7.33661493e-17f * g2 + -1.32287566f * g3 + -1.8753477e-16f * g4 + -1.70782513f * g5 + -1.21253394e-16f * g6;
        out3[node * 1792 + 512 + channel] = 1.19915417e-16f * g0 + -1.04119934e-16f * g1 + 4.45737309e-17f * g2 + 4.7819239e-16f * g3 + -0.540061725f * g4 + 5.93087191e-16f * g5 + -2.09165007f * g6;
        out3[node * 1792 + 768 + channel] = -1.61151678e-17f * g0 + -1.21537996e-16f * g1 + 2.1602469f * g2 + -3.77105422e-16f * g3 + 1.29803081e-16f * g4 + -3.66814977e-16f * g5 + 1.11968577e-16f * g6;
        out3[node * 1792 + 1024 + channel] = 4.30842745e-17f * g0 + 1.70782513f * g1 + -2.46895798e-16f * g2 + -3.62662667e-16f * g3 + 2.70240643e-16f * g4 + -3.10747419e-16f * g5 + -3.1250304e-17f * g6;
        out3[node * 1792 + 1280 + channel] = -5.81288948e-18f * g0 + 9.56230042e-18f * g1 + 1.39797718e-17f * g2 + 2.64575131f * g3 + 1.98851836e-16f * g4 + -2.37741639e-16f * g5 + 3.00545326e-17f * g6;
        out3[node * 1792 + 1536 + channel] = -9.19280599e-17f * g0 + -5.81952136e-17f * g1 + 2.7578418e-16f * g2 + 5.51568359e-16f * g3 + 2.1602469f * g4 + 1.98838766e-16f * g5 + -1.8385612e-16f * g6;
        break; }
    case 9: {
        const float g0 = in3[node * 1792 + 128 + channel];
        const float g1 = in3[node * 1792 + 384 + channel];
        const float g2 = in3[node * 1792 + 640 + channel];
        const float g3 = in3[node * 1792 + 896 + channel];
        const float g4 = in3[node * 1792 + 1152 + channel];
        const float g5 = in3[node * 1792 + 1408 + channel];
        const float g6 = in3[node * 1792 + 1664 + channel];
        out3[node * 1792 + 128 + channel] = -0.456435465f * g0 + 1.37950058e-17f * g1 + -0.353553391f * g2 + 1.69355302e-17f * g3 + 5.3362075e-18f * g4 + -1.79965091e-17f * g5 + 1.23102444e-17f * g6;
        out3[node * 1792 + 384 + channel] = 1.33747679e-17f * g0 + -1.3256712e-17f * g1 + 5.27165279e-17f * g2 + -0.288675135f * g3 + 3.21039153e-17f * g4 + -0.372677996f * g5 + 6.89677879e-17f * g6;
        out3[node * 1792 + 640 + channel] = -6.3605827e-17f * g0 + -2.22436498e-18f * g1 + -1.15646958e-17f * g3 + -0.11785113f * g4 + 2.47085095e-18f * g5 + -0.456435465f * g6;
        out3[node * 1792 + 896 + channel] = 1.1016285e-16f * g0 + -8.82410497e-18f * g1 + 0.471404521f * g2 + -7.47127048e-18f * g3 + 5.07936136e-18f * g4 + -1.94097114e-17f * g5 + 7.54760672e-18f * g6;
        out3[node * 1792 + 1152 + channel] = -2.00423487e-17f * g0 + 0.372677996f * g1 + 9.97852007e-19f * g3 + 5.64918386e-18f * g4 + -1.2830459e-17f * g5 + 4.70566367e-17f * g6;
        out3[node * 1792 + 1408 + channel] = 1.2436503e-17f * g0 + 1.27330232e-17f * g1 + -6.08816892e-17f * g2 + 0.577350269f * g3 + 2.35890195e-17f * g4 + 1.31648406e-17f * g5 + -2.13419497e-17f * g6;
        out3[node * 1792 + 1664 + channel] = 5.01943148e-17f * g0 + -4.3887246e-18f * g1 + -1.97433899e-18f * g2 + 7.57724106e-17f * g3 + 0.471404521f * g4 + 5.76299952e-17f * g5 + 1.26429714e-16f * g6;
        break; }
    }
}
