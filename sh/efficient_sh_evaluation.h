// Generated by Efficient Spherical Harmonics Evaluation.
// http://jcgt.org/published/0002/02/06/
// And adapted to match the same definition of 'order'.

#ifndef SH_EFFICIENT_SH_EVALUATION_H_
#define SH_EFFICIENT_SH_EVALUATION_H_

namespace sh {

// order 0
template <typename T>
void SHEval0(const T dX, const T dY, const T dZ, T *pSH)
{
   pSH[0] = static_cast<T>(0.2820947917738781);
}

// order 1
template <typename T>
void SHEval1(const T dX, const T dY, const T dZ, T *pSH)
{
   T c0,c1,s0,s1,tmpA;
   T dZ2 = dZ*dZ;

   pSH[0] = static_cast<T>(0.2820947917738781);
   pSH[2] = static_cast<T>(0.4886025119029199)*dZ;
   c0 = dX;
   s0 = dY;
   tmpA = static_cast<T>(-0.48860251190292);
   pSH[3] = tmpA*c0;
   pSH[1] = tmpA*s0;
}

// order 2
template <typename T>
void SHEval2(const T dX, const T dY, const T dZ, T *pSH)
{
   T c0,c1,s0,s1,tmpA,tmpB,tmpC;
   T dZ2 = dZ*dZ;

   pSH[0] = static_cast<T>(0.2820947917738781);
   pSH[2] = static_cast<T>(0.4886025119029199)*dZ;
   pSH[6] = static_cast<T>(0.9461746957575601)*dZ2 + static_cast<T>(-0.3153915652525201);
   c0 = dX;
   s0 = dY;

   tmpA = static_cast<T>(-0.48860251190292);
   pSH[3] = tmpA*c0;
   pSH[1] = tmpA*s0;
   tmpB = static_cast<T>(-1.092548430592079)*dZ;
   pSH[7] = tmpB*c0;
   pSH[5] = tmpB*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpC = static_cast<T>(0.5462742152960395);
   pSH[8] = tmpC*c1;
   pSH[4] = tmpC*s1;
}

// order 3
template <typename T>
void SHEval3(const T dX, const T dY, const T dZ, T *pSH)
{
   T c0,c1,s0,s1,tmpA,tmpB,tmpC;
   T dZ2 = dZ*dZ;

   pSH[0] = static_cast<T>(0.2820947917738781);
   pSH[2] = static_cast<T>(0.4886025119029199)*dZ;
   pSH[6] = static_cast<T>(0.9461746957575601)*dZ2 + static_cast<T>(-0.3153915652525201);
   pSH[12] = dZ*(static_cast<T>(1.865881662950577)*dZ2 + static_cast<T>(-1.119528997770346));
   c0 = dX;
   s0 = dY;

   tmpA = static_cast<T>(-0.48860251190292);
   pSH[3] = tmpA*c0;
   pSH[1] = tmpA*s0;
   tmpB = static_cast<T>(-1.092548430592079)*dZ;
   pSH[7] = tmpB*c0;
   pSH[5] = tmpB*s0;
   tmpC = static_cast<T>(-2.285228997322329)*dZ2 + static_cast<T>(0.4570457994644658);
   pSH[13] = tmpC*c0;
   pSH[11] = tmpC*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.5462742152960395);
   pSH[8] = tmpA*c1;
   pSH[4] = tmpA*s1;
   tmpB = static_cast<T>(1.445305721320277)*dZ;
   pSH[14] = tmpB*c1;
   pSH[10] = tmpB*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpC = static_cast<T>(-0.5900435899266435);
   pSH[15] = tmpC*c0;
   pSH[9] = tmpC*s0;
}

// order 4
template <typename T>
void SHEval4(const T dX, const T dY, const T dZ, T *pSH)
{
   T c0,c1,s0,s1,tmpA,tmpB,tmpC;
   T dZ2 = dZ*dZ;

   pSH[0] = static_cast<T>(0.2820947917738781);
   pSH[2] = static_cast<T>(0.4886025119029199)*dZ;
   pSH[6] = static_cast<T>(0.9461746957575601)*dZ2 + static_cast<T>(-0.3153915652525201);
   pSH[12] = dZ*(static_cast<T>(1.865881662950577)*dZ2 + static_cast<T>(-1.119528997770346));
   pSH[20] = static_cast<T>(1.984313483298443)*dZ*pSH[12] + static_cast<T>(-1.006230589874905)*pSH[6];
   c0 = dX;
   s0 = dY;

   tmpA = static_cast<T>(-0.48860251190292);
   pSH[3] = tmpA*c0;
   pSH[1] = tmpA*s0;
   tmpB = static_cast<T>(-1.092548430592079)*dZ;
   pSH[7] = tmpB*c0;
   pSH[5] = tmpB*s0;
   tmpC = static_cast<T>(-2.285228997322329)*dZ2 + static_cast<T>(0.4570457994644658);
   pSH[13] = tmpC*c0;
   pSH[11] = tmpC*s0;
   tmpA = dZ*(static_cast<T>(-4.683325804901025)*dZ2 + static_cast<T>(2.007139630671868));
   pSH[21] = tmpA*c0;
   pSH[19] = tmpA*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.5462742152960395);
   pSH[8] = tmpA*c1;
   pSH[4] = tmpA*s1;
   tmpB = static_cast<T>(1.445305721320277)*dZ;
   pSH[14] = tmpB*c1;
   pSH[10] = tmpB*s1;
   tmpC = static_cast<T>(3.31161143515146)*dZ2 + static_cast<T>(-0.47308734787878);
   pSH[22] = tmpC*c1;
   pSH[18] = tmpC*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpA = static_cast<T>(-0.5900435899266435);
   pSH[15] = tmpA*c0;
   pSH[9] = tmpA*s0;
   tmpB = static_cast<T>(-1.770130769779931)*dZ;
   pSH[23] = tmpB*c0;
   pSH[17] = tmpB*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpC = static_cast<T>(0.6258357354491763);
   pSH[24] = tmpC*c1;
   pSH[16] = tmpC*s1;
}

// order 6
template <typename T>
void SHEval5(const T dX, const T dY, const T dZ, T *pSH)
{
   T c0,c1,s0,s1,tmpA,tmpB,tmpC;
   T dZ2 = dZ*dZ;

   pSH[0] = static_cast<T>(0.2820947917738781);
   pSH[2] = static_cast<T>(0.4886025119029199)*dZ;
   pSH[6] = static_cast<T>(0.9461746957575601)*dZ2 + static_cast<T>(-0.3153915652525201);
   pSH[12] = dZ*(static_cast<T>(1.865881662950577)*dZ2 + static_cast<T>(-1.119528997770346));
   pSH[20] = static_cast<T>(1.984313483298443)*dZ*pSH[12] + static_cast<T>(-1.006230589874905)*pSH[6];
   pSH[30] = static_cast<T>(1.98997487421324)*dZ*pSH[20] + static_cast<T>(-1.002853072844814)*pSH[12];
   c0 = dX;
   s0 = dY;

   tmpA = static_cast<T>(-0.48860251190292);
   pSH[3] = tmpA*c0;
   pSH[1] = tmpA*s0;
   tmpB = static_cast<T>(-1.092548430592079)*dZ;
   pSH[7] = tmpB*c0;
   pSH[5] = tmpB*s0;
   tmpC = static_cast<T>(-2.285228997322329)*dZ2 + static_cast<T>(0.4570457994644658);
   pSH[13] = tmpC*c0;
   pSH[11] = tmpC*s0;
   tmpA = dZ*(static_cast<T>(-4.683325804901025)*dZ2 + static_cast<T>(2.007139630671868));
   pSH[21] = tmpA*c0;
   pSH[19] = tmpA*s0;
   tmpB = static_cast<T>(2.03100960115899)*dZ*tmpA + static_cast<T>(-0.991031208965115)*tmpC;
   pSH[31] = tmpB*c0;
   pSH[29] = tmpB*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.5462742152960395);
   pSH[8] = tmpA*c1;
   pSH[4] = tmpA*s1;
   tmpB = static_cast<T>(1.445305721320277)*dZ;
   pSH[14] = tmpB*c1;
   pSH[10] = tmpB*s1;
   tmpC = static_cast<T>(3.31161143515146)*dZ2 + static_cast<T>(-0.47308734787878);
   pSH[22] = tmpC*c1;
   pSH[18] = tmpC*s1;
   tmpA = dZ*(static_cast<T>(7.190305177459987)*dZ2 + static_cast<T>(-2.396768392486662));
   pSH[32] = tmpA*c1;
   pSH[28] = tmpA*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpA = static_cast<T>(-0.5900435899266435);
   pSH[15] = tmpA*c0;
   pSH[9] = tmpA*s0;
   tmpB = static_cast<T>(-1.770130769779931)*dZ;
   pSH[23] = tmpB*c0;
   pSH[17] = tmpB*s0;
   tmpC = static_cast<T>(-4.403144694917254)*dZ2 + static_cast<T>(0.4892382994352505);
   pSH[33] = tmpC*c0;
   pSH[27] = tmpC*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.6258357354491763);
   pSH[24] = tmpA*c1;
   pSH[16] = tmpA*s1;
   tmpB = static_cast<T>(2.075662314881041)*dZ;
   pSH[34] = tmpB*c1;
   pSH[26] = tmpB*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpC = static_cast<T>(-0.6563820568401703);
   pSH[35] = tmpC*c0;
   pSH[25] = tmpC*s0;
}

// order 6
template <typename T>
void SHEval6(const T dX, const T dY, const T dZ, T *pSH)
{
   T c0,c1,s0,s1,tmpA,tmpB,tmpC;
   T dZ2 = dZ*dZ;

   pSH[0] = static_cast<T>(0.2820947917738781);
   pSH[2] = static_cast<T>(0.4886025119029199)*dZ;
   pSH[6] = static_cast<T>(0.9461746957575601)*dZ2 + static_cast<T>(-0.3153915652525201);
   pSH[12] = dZ*(static_cast<T>(1.865881662950577)*dZ2 + static_cast<T>(-1.119528997770346));
   pSH[20] = static_cast<T>(1.984313483298443)*dZ*pSH[12] + static_cast<T>(-1.006230589874905)*pSH[6];
   pSH[30] = static_cast<T>(1.98997487421324)*dZ*pSH[20] + static_cast<T>(-1.002853072844814)*pSH[12];
   pSH[42] = static_cast<T>(1.993043457183567)*dZ*pSH[30] + static_cast<T>(-1.001542020962219)*pSH[20];
   c0 = dX;
   s0 = dY;

   tmpA = static_cast<T>(-0.48860251190292);
   pSH[3] = tmpA*c0;
   pSH[1] = tmpA*s0;
   tmpB = static_cast<T>(-1.092548430592079)*dZ;
   pSH[7] = tmpB*c0;
   pSH[5] = tmpB*s0;
   tmpC = static_cast<T>(-2.285228997322329)*dZ2 + static_cast<T>(0.4570457994644658);
   pSH[13] = tmpC*c0;
   pSH[11] = tmpC*s0;
   tmpA = dZ*(static_cast<T>(-4.683325804901025)*dZ2 + static_cast<T>(2.007139630671868));
   pSH[21] = tmpA*c0;
   pSH[19] = tmpA*s0;
   tmpB = static_cast<T>(2.03100960115899)*dZ*tmpA + static_cast<T>(-0.991031208965115)*tmpC;
   pSH[31] = tmpB*c0;
   pSH[29] = tmpB*s0;
   tmpC = static_cast<T>(2.021314989237028)*dZ*tmpB + static_cast<T>(-0.9952267030562385)*tmpA;
   pSH[43] = tmpC*c0;
   pSH[41] = tmpC*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.5462742152960395);
   pSH[8] = tmpA*c1;
   pSH[4] = tmpA*s1;
   tmpB = static_cast<T>(1.445305721320277)*dZ;
   pSH[14] = tmpB*c1;
   pSH[10] = tmpB*s1;
   tmpC = static_cast<T>(3.31161143515146)*dZ2 + static_cast<T>(-0.47308734787878);
   pSH[22] = tmpC*c1;
   pSH[18] = tmpC*s1;
   tmpA = dZ*(static_cast<T>(7.190305177459987)*dZ2 + static_cast<T>(-2.396768392486662));
   pSH[32] = tmpA*c1;
   pSH[28] = tmpA*s1;
   tmpB = static_cast<T>(2.11394181566097)*dZ*tmpA + static_cast<T>(-0.9736101204623268)*tmpC;
   pSH[44] = tmpB*c1;
   pSH[40] = tmpB*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpA = static_cast<T>(-0.5900435899266435);
   pSH[15] = tmpA*c0;
   pSH[9] = tmpA*s0;
   tmpB = static_cast<T>(-1.770130769779931)*dZ;
   pSH[23] = tmpB*c0;
   pSH[17] = tmpB*s0;
   tmpC = static_cast<T>(-4.403144694917254)*dZ2 + static_cast<T>(0.4892382994352505);
   pSH[33] = tmpC*c0;
   pSH[27] = tmpC*s0;
   tmpA = dZ*(static_cast<T>(-10.13325785466416)*dZ2 + static_cast<T>(2.763615778544771));
   pSH[45] = tmpA*c0;
   pSH[39] = tmpA*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.6258357354491763);
   pSH[24] = tmpA*c1;
   pSH[16] = tmpA*s1;
   tmpB = static_cast<T>(2.075662314881041)*dZ;
   pSH[34] = tmpB*c1;
   pSH[26] = tmpB*s1;
   tmpC = static_cast<T>(5.550213908015966)*dZ2 + static_cast<T>(-0.5045649007287241);
   pSH[46] = tmpC*c1;
   pSH[38] = tmpC*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpA = static_cast<T>(-0.6563820568401703);
   pSH[35] = tmpA*c0;
   pSH[25] = tmpA*s0;
   tmpB = static_cast<T>(-2.366619162231753)*dZ;
   pSH[47] = tmpB*c0;
   pSH[37] = tmpB*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpC = static_cast<T>(0.6831841051919144);
   pSH[48] = tmpC*c1;
   pSH[36] = tmpC*s1;
}

// order 7
template <typename T>
void SHEval7(const T dX, const T dY, const T dZ, T *pSH)
{
   T c0,c1,s0,s1,tmpA,tmpB,tmpC;
   T dZ2 = dZ*dZ;

   pSH[0] = static_cast<T>(0.2820947917738781);
   pSH[2] = static_cast<T>(0.4886025119029199)*dZ;
   pSH[6] = static_cast<T>(0.9461746957575601)*dZ2 + static_cast<T>(-0.3153915652525201);
   pSH[12] = dZ*(static_cast<T>(1.865881662950577)*dZ2 + static_cast<T>(-1.119528997770346));
   pSH[20] = static_cast<T>(1.984313483298443)*dZ*pSH[12] + static_cast<T>(-1.006230589874905)*pSH[6];
   pSH[30] = static_cast<T>(1.98997487421324)*dZ*pSH[20] + static_cast<T>(-1.002853072844814)*pSH[12];
   pSH[42] = static_cast<T>(1.993043457183567)*dZ*pSH[30] + static_cast<T>(-1.001542020962219)*pSH[20];
   pSH[56] = static_cast<T>(1.994891434824135)*dZ*pSH[42] + static_cast<T>(-1.000927213921958)*pSH[30];
   c0 = dX;
   s0 = dY;

   tmpA = static_cast<T>(-0.48860251190292);
   pSH[3] = tmpA*c0;
   pSH[1] = tmpA*s0;
   tmpB = static_cast<T>(-1.092548430592079)*dZ;
   pSH[7] = tmpB*c0;
   pSH[5] = tmpB*s0;
   tmpC = static_cast<T>(-2.285228997322329)*dZ2 + static_cast<T>(0.4570457994644658);
   pSH[13] = tmpC*c0;
   pSH[11] = tmpC*s0;
   tmpA = dZ*(static_cast<T>(-4.683325804901025)*dZ2 + static_cast<T>(2.007139630671868));
   pSH[21] = tmpA*c0;
   pSH[19] = tmpA*s0;
   tmpB = static_cast<T>(2.03100960115899)*dZ*tmpA + static_cast<T>(-0.991031208965115)*tmpC;
   pSH[31] = tmpB*c0;
   pSH[29] = tmpB*s0;
   tmpC = static_cast<T>(2.021314989237028)*dZ*tmpB + static_cast<T>(-0.9952267030562385)*tmpA;
   pSH[43] = tmpC*c0;
   pSH[41] = tmpC*s0;
   tmpA = static_cast<T>(2.015564437074638)*dZ*tmpC + static_cast<T>(-0.9971550440218319)*tmpB;
   pSH[57] = tmpA*c0;
   pSH[55] = tmpA*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.5462742152960395);
   pSH[8] = tmpA*c1;
   pSH[4] = tmpA*s1;
   tmpB = static_cast<T>(1.445305721320277)*dZ;
   pSH[14] = tmpB*c1;
   pSH[10] = tmpB*s1;
   tmpC = static_cast<T>(3.31161143515146)*dZ2 + static_cast<T>(-0.47308734787878);
   pSH[22] = tmpC*c1;
   pSH[18] = tmpC*s1;
   tmpA = dZ*(static_cast<T>(7.190305177459987)*dZ2 + static_cast<T>(-2.396768392486662));
   pSH[32] = tmpA*c1;
   pSH[28] = tmpA*s1;
   tmpB = static_cast<T>(2.11394181566097)*dZ*tmpA + static_cast<T>(-0.9736101204623268)*tmpC;
   pSH[44] = tmpB*c1;
   pSH[40] = tmpB*s1;
   tmpC = static_cast<T>(2.081665999466133)*dZ*tmpB + static_cast<T>(-0.9847319278346618)*tmpA;
   pSH[58] = tmpC*c1;
   pSH[54] = tmpC*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpA = static_cast<T>(-0.5900435899266435);
   pSH[15] = tmpA*c0;
   pSH[9] = tmpA*s0;
   tmpB = static_cast<T>(-1.770130769779931)*dZ;
   pSH[23] = tmpB*c0;
   pSH[17] = tmpB*s0;
   tmpC = static_cast<T>(-4.403144694917254)*dZ2 + static_cast<T>(0.4892382994352505);
   pSH[33] = tmpC*c0;
   pSH[27] = tmpC*s0;
   tmpA = dZ*(static_cast<T>(-10.13325785466416)*dZ2 + static_cast<T>(2.763615778544771));
   pSH[45] = tmpA*c0;
   pSH[39] = tmpA*s0;
   tmpB = static_cast<T>(2.207940216581962)*dZ*tmpA + static_cast<T>(-0.959403223600247)*tmpC;
   pSH[59] = tmpB*c0;
   pSH[53] = tmpB*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.6258357354491763);
   pSH[24] = tmpA*c1;
   pSH[16] = tmpA*s1;
   tmpB = static_cast<T>(2.075662314881041)*dZ;
   pSH[34] = tmpB*c1;
   pSH[26] = tmpB*s1;
   tmpC = static_cast<T>(5.550213908015966)*dZ2 + static_cast<T>(-0.5045649007287241);
   pSH[46] = tmpC*c1;
   pSH[38] = tmpC*s1;
   tmpA = dZ*(static_cast<T>(13.49180504672677)*dZ2 + static_cast<T>(-3.113493472321562));
   pSH[60] = tmpA*c1;
   pSH[52] = tmpA*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpA = static_cast<T>(-0.6563820568401703);
   pSH[35] = tmpA*c0;
   pSH[25] = tmpA*s0;
   tmpB = static_cast<T>(-2.366619162231753)*dZ;
   pSH[47] = tmpB*c0;
   pSH[37] = tmpB*s0;
   tmpC = static_cast<T>(-6.745902523363385)*dZ2 + static_cast<T>(0.5189155787202604);
   pSH[61] = tmpC*c0;
   pSH[51] = tmpC*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.6831841051919144);
   pSH[48] = tmpA*c1;
   pSH[36] = tmpA*s1;
   tmpB = static_cast<T>(2.645960661801901)*dZ;
   pSH[62] = tmpB*c1;
   pSH[50] = tmpB*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpC = static_cast<T>(-0.7071627325245963);
   pSH[63] = tmpC*c0;
   pSH[49] = tmpC*s0;
}

// order 8
template <typename T>
void SHEval8(const T dX, const T dY, const T dZ, T *pSH)
{
   T c0,c1,s0,s1,tmpA,tmpB,tmpC;
   T dZ2 = dZ*dZ;

   pSH[0] = static_cast<T>(0.2820947917738781);
   pSH[2] = static_cast<T>(0.4886025119029199)*dZ;
   pSH[6] = static_cast<T>(0.9461746957575601)*dZ2 + static_cast<T>(-0.3153915652525201);
   pSH[12] = dZ*(static_cast<T>(1.865881662950577)*dZ2 + static_cast<T>(-1.119528997770346));
   pSH[20] = static_cast<T>(1.984313483298443)*dZ*pSH[12] + static_cast<T>(-1.006230589874905)*pSH[6];
   pSH[30] = static_cast<T>(1.98997487421324)*dZ*pSH[20] + static_cast<T>(-1.002853072844814)*pSH[12];
   pSH[42] = static_cast<T>(1.993043457183567)*dZ*pSH[30] + static_cast<T>(-1.001542020962219)*pSH[20];
   pSH[56] = static_cast<T>(1.994891434824135)*dZ*pSH[42] + static_cast<T>(-1.000927213921958)*pSH[30];
   pSH[72] = static_cast<T>(1.996089927833914)*dZ*pSH[56] + static_cast<T>(-1.000600781069515)*pSH[42];
   c0 = dX;
   s0 = dY;

   tmpA = static_cast<T>(-0.48860251190292);
   pSH[3] = tmpA*c0;
   pSH[1] = tmpA*s0;
   tmpB = static_cast<T>(-1.092548430592079)*dZ;
   pSH[7] = tmpB*c0;
   pSH[5] = tmpB*s0;
   tmpC = static_cast<T>(-2.285228997322329)*dZ2 + static_cast<T>(0.4570457994644658);
   pSH[13] = tmpC*c0;
   pSH[11] = tmpC*s0;
   tmpA = dZ*(static_cast<T>(-4.683325804901025)*dZ2 + static_cast<T>(2.007139630671868));
   pSH[21] = tmpA*c0;
   pSH[19] = tmpA*s0;
   tmpB = static_cast<T>(2.03100960115899)*dZ*tmpA + static_cast<T>(-0.991031208965115)*tmpC;
   pSH[31] = tmpB*c0;
   pSH[29] = tmpB*s0;
   tmpC = static_cast<T>(2.021314989237028)*dZ*tmpB + static_cast<T>(-0.9952267030562385)*tmpA;
   pSH[43] = tmpC*c0;
   pSH[41] = tmpC*s0;
   tmpA = static_cast<T>(2.015564437074638)*dZ*tmpC + static_cast<T>(-0.9971550440218319)*tmpB;
   pSH[57] = tmpA*c0;
   pSH[55] = tmpA*s0;
   tmpB = static_cast<T>(2.011869540407391)*dZ*tmpA + static_cast<T>(-0.9981668178901745)*tmpC;
   pSH[73] = tmpB*c0;
   pSH[71] = tmpB*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.5462742152960395);
   pSH[8] = tmpA*c1;
   pSH[4] = tmpA*s1;
   tmpB = static_cast<T>(1.445305721320277)*dZ;
   pSH[14] = tmpB*c1;
   pSH[10] = tmpB*s1;
   tmpC = static_cast<T>(3.31161143515146)*dZ2 + static_cast<T>(-0.47308734787878);
   pSH[22] = tmpC*c1;
   pSH[18] = tmpC*s1;
   tmpA = dZ*(static_cast<T>(7.190305177459987)*dZ2 + static_cast<T>(-2.396768392486662));
   pSH[32] = tmpA*c1;
   pSH[28] = tmpA*s1;
   tmpB = static_cast<T>(2.11394181566097)*dZ*tmpA + static_cast<T>(-0.9736101204623268)*tmpC;
   pSH[44] = tmpB*c1;
   pSH[40] = tmpB*s1;
   tmpC = static_cast<T>(2.081665999466133)*dZ*tmpB + static_cast<T>(-0.9847319278346618)*tmpA;
   pSH[58] = tmpC*c1;
   pSH[54] = tmpC*s1;
   tmpA = static_cast<T>(2.06155281280883)*dZ*tmpC + static_cast<T>(-0.9903379376602873)*tmpB;
   pSH[74] = tmpA*c1;
   pSH[70] = tmpA*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpA = static_cast<T>(-0.5900435899266435);
   pSH[15] = tmpA*c0;
   pSH[9] = tmpA*s0;
   tmpB = static_cast<T>(-1.770130769779931)*dZ;
   pSH[23] = tmpB*c0;
   pSH[17] = tmpB*s0;
   tmpC = static_cast<T>(-4.403144694917254)*dZ2 + static_cast<T>(0.4892382994352505);
   pSH[33] = tmpC*c0;
   pSH[27] = tmpC*s0;
   tmpA = dZ*(static_cast<T>(-10.13325785466416)*dZ2 + static_cast<T>(2.763615778544771));
   pSH[45] = tmpA*c0;
   pSH[39] = tmpA*s0;
   tmpB = static_cast<T>(2.207940216581962)*dZ*tmpA + static_cast<T>(-0.959403223600247)*tmpC;
   pSH[59] = tmpB*c0;
   pSH[53] = tmpB*s0;
   tmpC = static_cast<T>(2.15322168769582)*dZ*tmpB + static_cast<T>(-0.9752173865600178)*tmpA;
   pSH[75] = tmpC*c0;
   pSH[69] = tmpC*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.6258357354491763);
   pSH[24] = tmpA*c1;
   pSH[16] = tmpA*s1;
   tmpB = static_cast<T>(2.075662314881041)*dZ;
   pSH[34] = tmpB*c1;
   pSH[26] = tmpB*s1;
   tmpC = static_cast<T>(5.550213908015966)*dZ2 + static_cast<T>(-0.5045649007287241);
   pSH[46] = tmpC*c1;
   pSH[38] = tmpC*s1;
   tmpA = dZ*(static_cast<T>(13.49180504672677)*dZ2 + static_cast<T>(-3.113493472321562));
   pSH[60] = tmpA*c1;
   pSH[52] = tmpA*s1;
   tmpB = static_cast<T>(2.304886114323221)*dZ*tmpA + static_cast<T>(-0.9481763873554654)*tmpC;
   pSH[76] = tmpB*c1;
   pSH[68] = tmpB*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpA = static_cast<T>(-0.6563820568401703);
   pSH[35] = tmpA*c0;
   pSH[25] = tmpA*s0;
   tmpB = static_cast<T>(-2.366619162231753)*dZ;
   pSH[47] = tmpB*c0;
   pSH[37] = tmpB*s0;
   tmpC = static_cast<T>(-6.745902523363385)*dZ2 + static_cast<T>(0.5189155787202604);
   pSH[61] = tmpC*c0;
   pSH[51] = tmpC*s0;
   tmpA = dZ*(static_cast<T>(-17.24955311049054)*dZ2 + static_cast<T>(3.449910622098108));
   pSH[77] = tmpA*c0;
   pSH[67] = tmpA*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.6831841051919144);
   pSH[48] = tmpA*c1;
   pSH[36] = tmpA*s1;
   tmpB = static_cast<T>(2.645960661801901)*dZ;
   pSH[62] = tmpB*c1;
   pSH[50] = tmpB*s1;
   tmpC = static_cast<T>(7.984991490893139)*dZ2 + static_cast<T>(-0.5323327660595426);
   pSH[78] = tmpC*c1;
   pSH[66] = tmpC*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpA = static_cast<T>(-0.7071627325245963);
   pSH[63] = tmpA*c0;
   pSH[49] = tmpA*s0;
   tmpB = static_cast<T>(-2.91570664069932)*dZ;
   pSH[79] = tmpB*c0;
   pSH[65] = tmpB*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpC = static_cast<T>(0.72892666017483);
   pSH[80] = tmpC*c1;
   pSH[64] = tmpC*s1;
}

// order 9
template <typename T>
void SHEval9(const T dX, const T dY, const T dZ, T *pSH)
{
   T c0,c1,s0,s1,tmpA,tmpB,tmpC;
   T dZ2 = dZ*dZ;

   pSH[0] = static_cast<T>(0.2820947917738781);
   pSH[2] = static_cast<T>(0.4886025119029199)*dZ;
   pSH[6] = static_cast<T>(0.9461746957575601)*dZ2 + static_cast<T>(-0.3153915652525201);
   pSH[12] = dZ*(static_cast<T>(1.865881662950577)*dZ2 + static_cast<T>(-1.119528997770346));
   pSH[20] = static_cast<T>(1.984313483298443)*dZ*pSH[12] + static_cast<T>(-1.006230589874905)*pSH[6];
   pSH[30] = static_cast<T>(1.98997487421324)*dZ*pSH[20] + static_cast<T>(-1.002853072844814)*pSH[12];
   pSH[42] = static_cast<T>(1.993043457183567)*dZ*pSH[30] + static_cast<T>(-1.001542020962219)*pSH[20];
   pSH[56] = static_cast<T>(1.994891434824135)*dZ*pSH[42] + static_cast<T>(-1.000927213921958)*pSH[30];
   pSH[72] = static_cast<T>(1.996089927833914)*dZ*pSH[56] + static_cast<T>(-1.000600781069515)*pSH[42];
   pSH[90] = static_cast<T>(1.996911195067937)*dZ*pSH[72] + static_cast<T>(-1.000411437993134)*pSH[56];
   c0 = dX;
   s0 = dY;

   tmpA = static_cast<T>(-0.48860251190292);
   pSH[3] = tmpA*c0;
   pSH[1] = tmpA*s0;
   tmpB = static_cast<T>(-1.092548430592079)*dZ;
   pSH[7] = tmpB*c0;
   pSH[5] = tmpB*s0;
   tmpC = static_cast<T>(-2.285228997322329)*dZ2 + static_cast<T>(0.4570457994644658);
   pSH[13] = tmpC*c0;
   pSH[11] = tmpC*s0;
   tmpA = dZ*(static_cast<T>(-4.683325804901025)*dZ2 + static_cast<T>(2.007139630671868));
   pSH[21] = tmpA*c0;
   pSH[19] = tmpA*s0;
   tmpB = static_cast<T>(2.03100960115899)*dZ*tmpA + static_cast<T>(-0.991031208965115)*tmpC;
   pSH[31] = tmpB*c0;
   pSH[29] = tmpB*s0;
   tmpC = static_cast<T>(2.021314989237028)*dZ*tmpB + static_cast<T>(-0.9952267030562385)*tmpA;
   pSH[43] = tmpC*c0;
   pSH[41] = tmpC*s0;
   tmpA = static_cast<T>(2.015564437074638)*dZ*tmpC + static_cast<T>(-0.9971550440218319)*tmpB;
   pSH[57] = tmpA*c0;
   pSH[55] = tmpA*s0;
   tmpB = static_cast<T>(2.011869540407391)*dZ*tmpA + static_cast<T>(-0.9981668178901745)*tmpC;
   pSH[73] = tmpB*c0;
   pSH[71] = tmpB*s0;
   tmpC = static_cast<T>(2.009353129741012)*dZ*tmpB + static_cast<T>(-0.9987492177719088)*tmpA;
   pSH[91] = tmpC*c0;
   pSH[89] = tmpC*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.5462742152960395);
   pSH[8] = tmpA*c1;
   pSH[4] = tmpA*s1;
   tmpB = static_cast<T>(1.445305721320277)*dZ;
   pSH[14] = tmpB*c1;
   pSH[10] = tmpB*s1;
   tmpC = static_cast<T>(3.31161143515146)*dZ2 + static_cast<T>(-0.47308734787878);
   pSH[22] = tmpC*c1;
   pSH[18] = tmpC*s1;
   tmpA = dZ*(static_cast<T>(7.190305177459987)*dZ2 + static_cast<T>(-2.396768392486662));
   pSH[32] = tmpA*c1;
   pSH[28] = tmpA*s1;
   tmpB = static_cast<T>(2.11394181566097)*dZ*tmpA + static_cast<T>(-0.9736101204623268)*tmpC;
   pSH[44] = tmpB*c1;
   pSH[40] = tmpB*s1;
   tmpC = static_cast<T>(2.081665999466133)*dZ*tmpB + static_cast<T>(-0.9847319278346618)*tmpA;
   pSH[58] = tmpC*c1;
   pSH[54] = tmpC*s1;
   tmpA = static_cast<T>(2.06155281280883)*dZ*tmpC + static_cast<T>(-0.9903379376602873)*tmpB;
   pSH[74] = tmpA*c1;
   pSH[70] = tmpA*s1;
   tmpB = static_cast<T>(2.048122358357819)*dZ*tmpA + static_cast<T>(-0.9934852726704042)*tmpC;
   pSH[92] = tmpB*c1;
   pSH[88] = tmpB*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpA = static_cast<T>(-0.5900435899266435);
   pSH[15] = tmpA*c0;
   pSH[9] = tmpA*s0;
   tmpB = static_cast<T>(-1.770130769779931)*dZ;
   pSH[23] = tmpB*c0;
   pSH[17] = tmpB*s0;
   tmpC = static_cast<T>(-4.403144694917254)*dZ2 + static_cast<T>(0.4892382994352505);
   pSH[33] = tmpC*c0;
   pSH[27] = tmpC*s0;
   tmpA = dZ*(static_cast<T>(-10.13325785466416)*dZ2 + static_cast<T>(2.763615778544771));
   pSH[45] = tmpA*c0;
   pSH[39] = tmpA*s0;
   tmpB = static_cast<T>(2.207940216581962)*dZ*tmpA + static_cast<T>(-0.959403223600247)*tmpC;
   pSH[59] = tmpB*c0;
   pSH[53] = tmpB*s0;
   tmpC = static_cast<T>(2.15322168769582)*dZ*tmpB + static_cast<T>(-0.9752173865600178)*tmpA;
   pSH[75] = tmpC*c0;
   pSH[69] = tmpC*s0;
   tmpA = static_cast<T>(2.118044171189805)*dZ*tmpC + static_cast<T>(-0.9836628449792094)*tmpB;
   pSH[93] = tmpA*c0;
   pSH[87] = tmpA*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.6258357354491763);
   pSH[24] = tmpA*c1;
   pSH[16] = tmpA*s1;
   tmpB = static_cast<T>(2.075662314881041)*dZ;
   pSH[34] = tmpB*c1;
   pSH[26] = tmpB*s1;
   tmpC = static_cast<T>(5.550213908015966)*dZ2 + static_cast<T>(-0.5045649007287241);
   pSH[46] = tmpC*c1;
   pSH[38] = tmpC*s1;
   tmpA = dZ*(static_cast<T>(13.49180504672677)*dZ2 + static_cast<T>(-3.113493472321562));
   pSH[60] = tmpA*c1;
   pSH[52] = tmpA*s1;
   tmpB = static_cast<T>(2.304886114323221)*dZ*tmpA + static_cast<T>(-0.9481763873554654)*tmpC;
   pSH[76] = tmpB*c1;
   pSH[68] = tmpB*s1;
   tmpC = static_cast<T>(2.229177150706235)*dZ*tmpB + static_cast<T>(-0.9671528397231821)*tmpA;
   pSH[94] = tmpC*c1;
   pSH[86] = tmpC*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpA = static_cast<T>(-0.6563820568401703);
   pSH[35] = tmpA*c0;
   pSH[25] = tmpA*s0;
   tmpB = static_cast<T>(-2.366619162231753)*dZ;
   pSH[47] = tmpB*c0;
   pSH[37] = tmpB*s0;
   tmpC = static_cast<T>(-6.745902523363385)*dZ2 + static_cast<T>(0.5189155787202604);
   pSH[61] = tmpC*c0;
   pSH[51] = tmpC*s0;
   tmpA = dZ*(static_cast<T>(-17.24955311049054)*dZ2 + static_cast<T>(3.449910622098108));
   pSH[77] = tmpA*c0;
   pSH[67] = tmpA*s0;
   tmpB = static_cast<T>(2.401636346922062)*dZ*tmpA + static_cast<T>(-0.9392246042043708)*tmpC;
   pSH[95] = tmpB*c0;
   pSH[85] = tmpB*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.6831841051919144);
   pSH[48] = tmpA*c1;
   pSH[36] = tmpA*s1;
   tmpB = static_cast<T>(2.645960661801901)*dZ;
   pSH[62] = tmpB*c1;
   pSH[50] = tmpB*s1;
   tmpC = static_cast<T>(7.984991490893139)*dZ2 + static_cast<T>(-0.5323327660595426);
   pSH[78] = tmpC*c1;
   pSH[66] = tmpC*s1;
   tmpA = dZ*(static_cast<T>(21.39289019090864)*dZ2 + static_cast<T>(-3.775215916042701));
   pSH[96] = tmpA*c1;
   pSH[84] = tmpA*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpA = static_cast<T>(-0.7071627325245963);
   pSH[63] = tmpA*c0;
   pSH[49] = tmpA*s0;
   tmpB = static_cast<T>(-2.91570664069932)*dZ;
   pSH[79] = tmpB*c0;
   pSH[65] = tmpB*s0;
   tmpC = static_cast<T>(-9.263393182848905)*dZ2 + static_cast<T>(0.5449054813440533);
   pSH[97] = tmpC*c0;
   pSH[83] = tmpC*s0;
   c1 = dX*c0 - dY*s0;
   s1 = dX*s0 + dY*c0;

   tmpA = static_cast<T>(0.72892666017483);
   pSH[80] = tmpA*c1;
   pSH[64] = tmpA*s1;
   tmpB = static_cast<T>(3.177317648954698)*dZ;
   pSH[98] = tmpB*c1;
   pSH[82] = tmpB*s1;
   c0 = dX*c1 - dY*s1;
   s0 = dX*s1 + dY*c1;

   tmpC = static_cast<T>(-0.7489009518531884);
   pSH[99] = tmpC*c0;
   pSH[81] = tmpC*s0;
}

template <typename T>
void (*SHEval[])(const T dX, const T dY, const T dZ, T *pSH) =
    { SHEval0,
      SHEval1,
      SHEval2,
      SHEval3,
      SHEval4,
      SHEval5,
      SHEval6,
      SHEval7,
      SHEval8,
      SHEval9 };

}  // namespace sh

#endif  // SH_EFFICIENT_SH_EVALUATION_H_