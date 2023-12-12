#include <sip30intrin.h>

#define TAR_ADDR_WARP(addr, ss) (((addr)) | ((((addr) + (ss)) << 16)))
#define TAR_OFF_WARP(offset1, offset2) (((offset1) << 16) | ((offset2)&0xffff))

#define LOAD_SMR_MODE17_BF16_ROW(smr, offset)  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 0);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 1);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 2);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 3);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 4);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 5);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 6);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 7);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 8);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 9);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 10); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 11); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 12); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 13); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 14); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 15); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 16); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 17); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 18); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 19); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 20); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 21); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 22); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 23); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 24); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 25); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 26); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 27); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 28); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 29); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, rt_off0, 30); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr, rt_base, offset, 31);

#define LOAD_LHS(offset)  \
  vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr10 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr11 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr12 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr13 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr14 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr15 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr16 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr17 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr18 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr19 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr20 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr21 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr22 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr23 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr24 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr25 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr26 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr27 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr28 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr29 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr30 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr31 = __dtu_s_tvld_itar(lt_base, offset);

#define VMM_MODE17_BF16(pos, smr)  \
  qacc[0+pos] = __dtu_m_vmm2_mode17_bf16(qacc[0+pos], vr0, smr);     \
  qacc[1+pos] = __dtu_m_vmm2_mode17_bf16(qacc[1+pos], vr1, smr);     \
  qacc[2+pos] = __dtu_m_vmm2_mode17_bf16(qacc[2+pos], vr2, smr);     \
  qacc[3+pos] = __dtu_m_vmm2_mode17_bf16(qacc[3+pos], vr3, smr);     \
  qacc[4+pos] = __dtu_m_vmm2_mode17_bf16(qacc[4+pos], vr4, smr);     \
  qacc[5+pos] = __dtu_m_vmm2_mode17_bf16(qacc[5+pos], vr5, smr);     \
  qacc[6+pos] = __dtu_m_vmm2_mode17_bf16(qacc[6+pos], vr6, smr);     \
  qacc[7+pos] = __dtu_m_vmm2_mode17_bf16(qacc[7+pos], vr7, smr);     \
  qacc[8+pos] = __dtu_m_vmm2_mode17_bf16(qacc[8+pos], vr8, smr);     \
  qacc[9+pos] = __dtu_m_vmm2_mode17_bf16(qacc[9+pos], vr9, smr);     \
  qacc[10+pos] = __dtu_m_vmm2_mode17_bf16(qacc[10+pos], vr10, smr);  \
  qacc[11+pos] = __dtu_m_vmm2_mode17_bf16(qacc[11+pos], vr11, smr);  \
  qacc[12+pos] = __dtu_m_vmm2_mode17_bf16(qacc[12+pos], vr12, smr);  \
  qacc[13+pos] = __dtu_m_vmm2_mode17_bf16(qacc[13+pos], vr13, smr);  \
  qacc[14+pos] = __dtu_m_vmm2_mode17_bf16(qacc[14+pos], vr14, smr);  \
  qacc[15+pos] = __dtu_m_vmm2_mode17_bf16(qacc[15+pos], vr15, smr);  \
  qacc[16+pos] = __dtu_m_vmm2_mode17_bf16(qacc[16+pos], vr16, smr);  \
  qacc[17+pos] = __dtu_m_vmm2_mode17_bf16(qacc[17+pos], vr17, smr);  \
  qacc[18+pos] = __dtu_m_vmm2_mode17_bf16(qacc[18+pos], vr18, smr);  \
  qacc[19+pos] = __dtu_m_vmm2_mode17_bf16(qacc[19+pos], vr19, smr);  \
  qacc[20+pos] = __dtu_m_vmm2_mode17_bf16(qacc[20+pos], vr20, smr);  \
  qacc[21+pos] = __dtu_m_vmm2_mode17_bf16(qacc[21+pos], vr21, smr);  \
  qacc[22+pos] = __dtu_m_vmm2_mode17_bf16(qacc[22+pos], vr22, smr);  \
  qacc[23+pos] = __dtu_m_vmm2_mode17_bf16(qacc[23+pos], vr23, smr);  \
  qacc[24+pos] = __dtu_m_vmm2_mode17_bf16(qacc[24+pos], vr24, smr);  \
  qacc[25+pos] = __dtu_m_vmm2_mode17_bf16(qacc[25+pos], vr25, smr);  \
  qacc[26+pos] = __dtu_m_vmm2_mode17_bf16(qacc[26+pos], vr26, smr);  \
  qacc[27+pos] = __dtu_m_vmm2_mode17_bf16(qacc[27+pos], vr27, smr);  \
  qacc[28+pos] = __dtu_m_vmm2_mode17_bf16(qacc[28+pos], vr28, smr);  \
  qacc[29+pos] = __dtu_m_vmm2_mode17_bf16(qacc[29+pos], vr29, smr);  \
  qacc[30+pos] = __dtu_m_vmm2_mode17_bf16(qacc[30+pos], vr30, smr);  \
  qacc[31+pos] = __dtu_m_vmm2_mode17_bf16(qacc[31+pos], vr31, smr);
#define LOAD_OUT(offset) \
  qacc[64] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[65] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[66] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[67] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[68] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[69] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[70] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[71] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[72] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[73] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[74] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[75] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[76] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[77] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[78] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[79] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[80] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[81] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[82] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[83] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[84] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[85] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[86] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[87] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[88] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[89] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[90] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[91] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[92] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[93] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[94] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[95] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[96] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[97] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[98] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[99] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[100] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[101] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[102] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[103] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[104] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[105] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[106] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[107] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[108] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[109] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[110] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[111] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[112] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[113] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[114] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[115] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[116] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[117] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[118] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[119] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[120] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[121] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[122] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[123] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[124] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[125] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[126] = __dtu_l_tvldqa_f32_qa(bt_base, bt_off0); \
  qacc[127] = __dtu_l_tvldqa_f32_qa(bt_base, offset);

#define STORE_OUT(offset)  \
  __dtu_v_tvstda_f32_dual(c_dacc[0], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[1], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[2], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[3], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[4], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[5], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[6], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[7], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[8], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[9], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[10], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[11], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[12], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[13], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[14], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[15], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[16], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[17], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[18], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[19], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[20], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[21], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[22], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[23], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[24], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[25], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[26], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[27], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[28], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[29], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[30], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[31], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[32], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[33], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[34], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[35], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[36], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[37], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[38], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[39], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[40], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[41], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[42], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[43], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[44], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[45], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[46], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[47], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[48], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[49], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[50], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[51], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[52], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[53], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[54], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[55], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[56], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[57], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[58], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[59], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[60], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[61], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[62], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[63], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[64], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[65], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[66], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[67], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[68], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[69], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[70], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[71], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[72], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[73], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[74], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[75], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[76], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[77], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[78], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[79], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[80], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[81], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[82], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[83], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[84], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[85], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[86], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[87], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[88], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[89], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[90], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[91], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[92], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[93], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[94], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[95], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[96], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[97], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[98], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[99], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[100], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[101], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[102], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[103], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[104], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[105], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[106], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[107], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[108], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[109], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[110], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[111], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[112], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[113], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[114], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[115], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[116], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[117], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[118], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[119], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[120], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[121], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[122], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[123], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[124], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[125], ot_base, ot_off3); \
  __dtu_v_tvstda_f32_dual(c_dacc[126], ot_base, ot_off0); \
  __dtu_v_tvstda_f32_dual(c_dacc[127], ot_base, offset);

#define MOP_ADD_FP32 \
  qacc[0] = __dtu_m_mop_add_f32_qa(qacc[64], qacc[0]); \
  qacc[1] = __dtu_m_mop_add_f32_qa(qacc[65], qacc[1]); \
  qacc[2] = __dtu_m_mop_add_f32_qa(qacc[66], qacc[2]); \
  qacc[3] = __dtu_m_mop_add_f32_qa(qacc[67], qacc[3]); \
  qacc[4] = __dtu_m_mop_add_f32_qa(qacc[68], qacc[4]); \
  qacc[5] = __dtu_m_mop_add_f32_qa(qacc[69], qacc[5]); \
  qacc[6] = __dtu_m_mop_add_f32_qa(qacc[70], qacc[6]); \
  qacc[7] = __dtu_m_mop_add_f32_qa(qacc[71], qacc[7]); \
  qacc[8] = __dtu_m_mop_add_f32_qa(qacc[72], qacc[8]); \
  qacc[9] = __dtu_m_mop_add_f32_qa(qacc[73], qacc[9]); \
  qacc[10] = __dtu_m_mop_add_f32_qa(qacc[74], qacc[10]); \
  qacc[11] = __dtu_m_mop_add_f32_qa(qacc[75], qacc[11]); \
  qacc[12] = __dtu_m_mop_add_f32_qa(qacc[76], qacc[12]); \
  qacc[13] = __dtu_m_mop_add_f32_qa(qacc[77], qacc[13]); \
  qacc[14] = __dtu_m_mop_add_f32_qa(qacc[78], qacc[14]); \
  qacc[15] = __dtu_m_mop_add_f32_qa(qacc[79], qacc[15]); \
  qacc[16] = __dtu_m_mop_add_f32_qa(qacc[80], qacc[16]); \
  qacc[17] = __dtu_m_mop_add_f32_qa(qacc[81], qacc[17]); \
  qacc[18] = __dtu_m_mop_add_f32_qa(qacc[82], qacc[18]); \
  qacc[19] = __dtu_m_mop_add_f32_qa(qacc[83], qacc[19]); \
  qacc[20] = __dtu_m_mop_add_f32_qa(qacc[84], qacc[20]); \
  qacc[21] = __dtu_m_mop_add_f32_qa(qacc[85], qacc[21]); \
  qacc[22] = __dtu_m_mop_add_f32_qa(qacc[86], qacc[22]); \
  qacc[23] = __dtu_m_mop_add_f32_qa(qacc[87], qacc[23]); \
  qacc[24] = __dtu_m_mop_add_f32_qa(qacc[88], qacc[24]); \
  qacc[25] = __dtu_m_mop_add_f32_qa(qacc[89], qacc[25]); \
  qacc[26] = __dtu_m_mop_add_f32_qa(qacc[90], qacc[26]); \
  qacc[27] = __dtu_m_mop_add_f32_qa(qacc[91], qacc[27]); \
  qacc[28] = __dtu_m_mop_add_f32_qa(qacc[92], qacc[28]); \
  qacc[29] = __dtu_m_mop_add_f32_qa(qacc[93], qacc[29]); \
  qacc[30] = __dtu_m_mop_add_f32_qa(qacc[94], qacc[30]); \
  qacc[31] = __dtu_m_mop_add_f32_qa(qacc[95], qacc[31]); \
  qacc[32] = __dtu_m_mop_add_f32_qa(qacc[96], qacc[32]); \
  qacc[33] = __dtu_m_mop_add_f32_qa(qacc[97], qacc[33]); \
  qacc[34] = __dtu_m_mop_add_f32_qa(qacc[98], qacc[34]); \
  qacc[35] = __dtu_m_mop_add_f32_qa(qacc[99], qacc[35]); \
  qacc[36] = __dtu_m_mop_add_f32_qa(qacc[100], qacc[36]); \
  qacc[37] = __dtu_m_mop_add_f32_qa(qacc[101], qacc[37]); \
  qacc[38] = __dtu_m_mop_add_f32_qa(qacc[102], qacc[38]); \
  qacc[39] = __dtu_m_mop_add_f32_qa(qacc[103], qacc[39]); \
  qacc[40] = __dtu_m_mop_add_f32_qa(qacc[104], qacc[40]); \
  qacc[41] = __dtu_m_mop_add_f32_qa(qacc[105], qacc[41]); \
  qacc[42] = __dtu_m_mop_add_f32_qa(qacc[106], qacc[42]); \
  qacc[43] = __dtu_m_mop_add_f32_qa(qacc[107], qacc[43]); \
  qacc[44] = __dtu_m_mop_add_f32_qa(qacc[108], qacc[44]); \
  qacc[45] = __dtu_m_mop_add_f32_qa(qacc[109], qacc[45]); \
  qacc[46] = __dtu_m_mop_add_f32_qa(qacc[110], qacc[46]); \
  qacc[47] = __dtu_m_mop_add_f32_qa(qacc[111], qacc[47]); \
  qacc[48] = __dtu_m_mop_add_f32_qa(qacc[112], qacc[48]); \
  qacc[49] = __dtu_m_mop_add_f32_qa(qacc[113], qacc[49]); \
  qacc[50] = __dtu_m_mop_add_f32_qa(qacc[114], qacc[50]); \
  qacc[51] = __dtu_m_mop_add_f32_qa(qacc[115], qacc[51]); \
  qacc[52] = __dtu_m_mop_add_f32_qa(qacc[116], qacc[52]); \
  qacc[53] = __dtu_m_mop_add_f32_qa(qacc[117], qacc[53]); \
  qacc[54] = __dtu_m_mop_add_f32_qa(qacc[118], qacc[54]); \
  qacc[55] = __dtu_m_mop_add_f32_qa(qacc[119], qacc[55]); \
  qacc[56] = __dtu_m_mop_add_f32_qa(qacc[120], qacc[56]); \
  qacc[57] = __dtu_m_mop_add_f32_qa(qacc[121], qacc[57]); \
  qacc[58] = __dtu_m_mop_add_f32_qa(qacc[122], qacc[58]); \
  qacc[59] = __dtu_m_mop_add_f32_qa(qacc[123], qacc[59]); \
  qacc[60] = __dtu_m_mop_add_f32_qa(qacc[124], qacc[60]); \
  qacc[61] = __dtu_m_mop_add_f32_qa(qacc[125], qacc[61]); \
  qacc[62] = __dtu_m_mop_add_f32_qa(qacc[126], qacc[62]); \
  qacc[63] = __dtu_m_mop_add_f32_qa(qacc[127], qacc[63]);

#define QA2DA \
  c_dacc[0] = __dtu_extractqa2da(qacc[0], 0); \
  c_dacc[1] = __dtu_extractqa2da(qacc[0], 1); \
  c_dacc[2] = __dtu_extractqa2da(qacc[1], 0); \
  c_dacc[3] = __dtu_extractqa2da(qacc[1], 1); \
  c_dacc[4] = __dtu_extractqa2da(qacc[2], 0); \
  c_dacc[5] = __dtu_extractqa2da(qacc[2], 1); \
  c_dacc[6] = __dtu_extractqa2da(qacc[3], 0); \
  c_dacc[7] = __dtu_extractqa2da(qacc[3], 1); \
  c_dacc[8] = __dtu_extractqa2da(qacc[4], 0); \
  c_dacc[9] = __dtu_extractqa2da(qacc[4], 1); \
  c_dacc[10] = __dtu_extractqa2da(qacc[5], 0); \
  c_dacc[11] = __dtu_extractqa2da(qacc[5], 1); \
  c_dacc[12] = __dtu_extractqa2da(qacc[6], 0); \
  c_dacc[13] = __dtu_extractqa2da(qacc[6], 1); \
  c_dacc[14] = __dtu_extractqa2da(qacc[7], 0); \
  c_dacc[15] = __dtu_extractqa2da(qacc[7], 1); \
  c_dacc[16] = __dtu_extractqa2da(qacc[8], 0); \
  c_dacc[17] = __dtu_extractqa2da(qacc[8], 1); \
  c_dacc[18] = __dtu_extractqa2da(qacc[9], 0); \
  c_dacc[19] = __dtu_extractqa2da(qacc[9], 1); \
  c_dacc[20] = __dtu_extractqa2da(qacc[10], 0); \
  c_dacc[21] = __dtu_extractqa2da(qacc[10], 1); \
  c_dacc[22] = __dtu_extractqa2da(qacc[11], 0); \
  c_dacc[23] = __dtu_extractqa2da(qacc[11], 1); \
  c_dacc[24] = __dtu_extractqa2da(qacc[12], 0); \
  c_dacc[25] = __dtu_extractqa2da(qacc[12], 1); \
  c_dacc[26] = __dtu_extractqa2da(qacc[13], 0); \
  c_dacc[27] = __dtu_extractqa2da(qacc[13], 1); \
  c_dacc[28] = __dtu_extractqa2da(qacc[14], 0); \
  c_dacc[29] = __dtu_extractqa2da(qacc[14], 1); \
  c_dacc[30] = __dtu_extractqa2da(qacc[15], 0); \
  c_dacc[31] = __dtu_extractqa2da(qacc[15], 1); \
  c_dacc[32] = __dtu_extractqa2da(qacc[16], 0); \
  c_dacc[33] = __dtu_extractqa2da(qacc[16], 1); \
  c_dacc[34] = __dtu_extractqa2da(qacc[17], 0); \
  c_dacc[35] = __dtu_extractqa2da(qacc[17], 1); \
  c_dacc[36] = __dtu_extractqa2da(qacc[18], 0); \
  c_dacc[37] = __dtu_extractqa2da(qacc[18], 1); \
  c_dacc[38] = __dtu_extractqa2da(qacc[19], 0); \
  c_dacc[39] = __dtu_extractqa2da(qacc[19], 1); \
  c_dacc[40] = __dtu_extractqa2da(qacc[20], 0); \
  c_dacc[41] = __dtu_extractqa2da(qacc[20], 1); \
  c_dacc[42] = __dtu_extractqa2da(qacc[21], 0); \
  c_dacc[43] = __dtu_extractqa2da(qacc[21], 1); \
  c_dacc[44] = __dtu_extractqa2da(qacc[22], 0); \
  c_dacc[45] = __dtu_extractqa2da(qacc[22], 1); \
  c_dacc[46] = __dtu_extractqa2da(qacc[23], 0); \
  c_dacc[47] = __dtu_extractqa2da(qacc[23], 1); \
  c_dacc[48] = __dtu_extractqa2da(qacc[24], 0); \
  c_dacc[49] = __dtu_extractqa2da(qacc[24], 1); \
  c_dacc[50] = __dtu_extractqa2da(qacc[25], 0); \
  c_dacc[51] = __dtu_extractqa2da(qacc[25], 1); \
  c_dacc[52] = __dtu_extractqa2da(qacc[26], 0); \
  c_dacc[53] = __dtu_extractqa2da(qacc[26], 1); \
  c_dacc[54] = __dtu_extractqa2da(qacc[27], 0); \
  c_dacc[55] = __dtu_extractqa2da(qacc[27], 1); \
  c_dacc[56] = __dtu_extractqa2da(qacc[28], 0); \
  c_dacc[57] = __dtu_extractqa2da(qacc[28], 1); \
  c_dacc[58] = __dtu_extractqa2da(qacc[29], 0); \
  c_dacc[59] = __dtu_extractqa2da(qacc[29], 1); \
  c_dacc[60] = __dtu_extractqa2da(qacc[30], 0); \
  c_dacc[61] = __dtu_extractqa2da(qacc[30], 1); \
  c_dacc[62] = __dtu_extractqa2da(qacc[31], 0); \
  c_dacc[63] = __dtu_extractqa2da(qacc[31], 1); \
  c_dacc[64] = __dtu_extractqa2da(qacc[32], 0); \
  c_dacc[65] = __dtu_extractqa2da(qacc[32], 1); \
  c_dacc[66] = __dtu_extractqa2da(qacc[33], 0); \
  c_dacc[67] = __dtu_extractqa2da(qacc[33], 1); \
  c_dacc[68] = __dtu_extractqa2da(qacc[34], 0); \
  c_dacc[69] = __dtu_extractqa2da(qacc[34], 1); \
  c_dacc[70] = __dtu_extractqa2da(qacc[35], 0); \
  c_dacc[71] = __dtu_extractqa2da(qacc[35], 1); \
  c_dacc[72] = __dtu_extractqa2da(qacc[36], 0); \
  c_dacc[73] = __dtu_extractqa2da(qacc[36], 1); \
  c_dacc[74] = __dtu_extractqa2da(qacc[37], 0); \
  c_dacc[75] = __dtu_extractqa2da(qacc[37], 1); \
  c_dacc[76] = __dtu_extractqa2da(qacc[38], 0); \
  c_dacc[77] = __dtu_extractqa2da(qacc[38], 1); \
  c_dacc[78] = __dtu_extractqa2da(qacc[39], 0); \
  c_dacc[79] = __dtu_extractqa2da(qacc[39], 1); \
  c_dacc[80] = __dtu_extractqa2da(qacc[40], 0); \
  c_dacc[81] = __dtu_extractqa2da(qacc[40], 1); \
  c_dacc[82] = __dtu_extractqa2da(qacc[41], 0); \
  c_dacc[83] = __dtu_extractqa2da(qacc[41], 1); \
  c_dacc[84] = __dtu_extractqa2da(qacc[42], 0); \
  c_dacc[85] = __dtu_extractqa2da(qacc[42], 1); \
  c_dacc[86] = __dtu_extractqa2da(qacc[43], 0); \
  c_dacc[87] = __dtu_extractqa2da(qacc[43], 1); \
  c_dacc[88] = __dtu_extractqa2da(qacc[44], 0); \
  c_dacc[89] = __dtu_extractqa2da(qacc[44], 1); \
  c_dacc[90] = __dtu_extractqa2da(qacc[45], 0); \
  c_dacc[91] = __dtu_extractqa2da(qacc[45], 1); \
  c_dacc[92] = __dtu_extractqa2da(qacc[46], 0); \
  c_dacc[93] = __dtu_extractqa2da(qacc[46], 1); \
  c_dacc[94] = __dtu_extractqa2da(qacc[47], 0); \
  c_dacc[95] = __dtu_extractqa2da(qacc[47], 1); \
  c_dacc[96] = __dtu_extractqa2da(qacc[48], 0); \
  c_dacc[97] = __dtu_extractqa2da(qacc[48], 1); \
  c_dacc[98] = __dtu_extractqa2da(qacc[49], 0); \
  c_dacc[99] = __dtu_extractqa2da(qacc[49], 1); \
  c_dacc[100] = __dtu_extractqa2da(qacc[50], 0); \
  c_dacc[101] = __dtu_extractqa2da(qacc[50], 1); \
  c_dacc[102] = __dtu_extractqa2da(qacc[51], 0); \
  c_dacc[103] = __dtu_extractqa2da(qacc[51], 1); \
  c_dacc[104] = __dtu_extractqa2da(qacc[52], 0); \
  c_dacc[105] = __dtu_extractqa2da(qacc[52], 1); \
  c_dacc[106] = __dtu_extractqa2da(qacc[53], 0); \
  c_dacc[107] = __dtu_extractqa2da(qacc[53], 1); \
  c_dacc[108] = __dtu_extractqa2da(qacc[54], 0); \
  c_dacc[109] = __dtu_extractqa2da(qacc[54], 1); \
  c_dacc[110] = __dtu_extractqa2da(qacc[55], 0); \
  c_dacc[111] = __dtu_extractqa2da(qacc[55], 1); \
  c_dacc[112] = __dtu_extractqa2da(qacc[56], 0); \
  c_dacc[113] = __dtu_extractqa2da(qacc[56], 1); \
  c_dacc[114] = __dtu_extractqa2da(qacc[57], 0); \
  c_dacc[115] = __dtu_extractqa2da(qacc[57], 1); \
  c_dacc[116] = __dtu_extractqa2da(qacc[58], 0); \
  c_dacc[117] = __dtu_extractqa2da(qacc[58], 1); \
  c_dacc[118] = __dtu_extractqa2da(qacc[59], 0); \
  c_dacc[119] = __dtu_extractqa2da(qacc[59], 1); \
  c_dacc[120] = __dtu_extractqa2da(qacc[60], 0); \
  c_dacc[121] = __dtu_extractqa2da(qacc[60], 1); \
  c_dacc[122] = __dtu_extractqa2da(qacc[61], 0); \
  c_dacc[123] = __dtu_extractqa2da(qacc[61], 1); \
  c_dacc[124] = __dtu_extractqa2da(qacc[62], 0); \
  c_dacc[125] = __dtu_extractqa2da(qacc[62], 1); \
  c_dacc[126] = __dtu_extractqa2da(qacc[63], 0); \
  c_dacc[127] = __dtu_extractqa2da(qacc[63], 1);

__attribute__((device, dtu_maxinum_vacc(1024))) extern "C" void
c_func_bfmatmul_general(int a_addr, int b_addr, int c_addr, int M, int N,
  int K) {
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2vab_m_d(0);

  int BPE = 2;
  smr_t smr0, smr1;
  v64i8 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;

  va16f32x4 qacc[128];
  va16bf16x2 c_dacc[128];

  auto k_unit = K >> 5;
  auto n_unit = N >> 6;
  auto on_unit = N >> 5;
  // vpt parallel in rhs
  int lt_addr = a_addr >> 6;
  int rt_addr = b_addr >> 7;
  int ot_addr = c_addr >> 7;
  int offset = 0;
  tar_t lt_base = __dtu_c_movsr2targ(TAR_ADDR_WARP(lt_addr, 0));
  offset = TAR_OFF_WARP(k_unit, k_unit);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1 - 63 * k_unit, 1 - 63 * k_unit);  // next k
  tar_t lt_off1 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1 - 64 * k_unit, 1 - 64 * k_unit);  //  new n
  tar_t lt_off2 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1, 1);  // end k end n new m
  tar_t lt_off3 = __dtu_c_movsr2tari(offset, lt_base);

  tar_t rt_base = __dtu_c_movsr2targ((rt_addr) | ((rt_addr) + 1) << 16);
  offset = TAR_OFF_WARP(n_unit, n_unit);
  tar_t rt_off0 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR_OFF_WARP(2 - (K - 1) * n_unit, 2 - (K - 1) * n_unit);
  tar_t rt_off1 = __dtu_c_movsr2tari(offset, rt_base);  // new n
  offset = TAR_OFF_WARP(2 - K * n_unit, 2 - K * n_unit);
  tar_t rt_off2 = __dtu_c_movsr2tari(offset, rt_base);  // new m

  auto bn_unit = N >> 6;
  int bt_addr = (c_addr >> 8) | ((c_addr >> 8) + 1) << 16;
  tar_t bt_base = __dtu_c_movsr2targ(bt_addr);
  offset = (bn_unit << 16) | bn_unit;
  tar_t bt_off0 = __dtu_c_movsr2tari(offset, bt_base);
  offset = (2 - 63 * bn_unit) & 0xffff;
  offset = (offset << 16) | offset;
  tar_t bt_off1 = __dtu_c_movsr2tari(offset, bt_base);
  offset = (2 << 16) | 2;
  tar_t bt_off2 = __dtu_c_movsr2tari(offset, bt_base);

  tar_t ot_base = __dtu_c_movsr2targ((ot_addr) | ((ot_addr) + 2) << 16);
  offset = TAR_OFF_WARP(1, 1);
  tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
  offset = TAR_OFF_WARP(3 - 63 * on_unit, 3 - 63 * on_unit);
  tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);  // new n
  offset = TAR_OFF_WARP(3, 3);
  tar_t ot_off2 = __dtu_c_movsr2tari(offset, ot_base);  // new m
  offset = TAR_OFF_WARP(on_unit - 1, on_unit - 1);
  tar_t ot_off3 = __dtu_c_movsr2tari(offset, ot_base);

  // k0n0
  LOAD_SMR_MODE17_BF16_ROW(smr0, rt_off0);
  // m0k0
  LOAD_LHS(lt_off0);
  int naccovr = 0x10001;
  __dtu_c_movsr2naccovr(naccovr);
  __dtu_c_movsr2vab_m_s2(0);
  int vab_shift = 0;
// fp16 vmm2 mode17: [64, 64] * [64, 128] = [64, 128]
#pragma clang loop unroll(full)
  for (int m = 0; m < M; m += 64) {
    for (int n = 0; n < N - 128; n += 128) {  // VPT PARA DIM
      __dtu_c_movsr2naccovr(naccovr);
      for (int k = 0; k < K - 64; k += 64) {
        // smr1
        LOAD_SMR_MODE17_BF16_ROW(smr1, rt_off0);
        // m0k0 * smr0
        VMM_MODE17_BF16(0, smr0);
        // m1k0
        LOAD_LHS(lt_off1);
        // m1k0 * smr0
        VMM_MODE17_BF16(32, smr0);

        // m0k1
        LOAD_LHS(lt_off0);
        // next k unit smr0
        LOAD_SMR_MODE17_BF16_ROW(smr0, rt_off0);
        __dtu_c_movsr2naccovr(0x1);
        // m0k1 * smr1
        VMM_MODE17_BF16(0, smr1);
        // m1k1
        LOAD_LHS(lt_off1);
        // m1k1 * smr1
        VMM_MODE17_BF16(32, smr1);
        // next k unit m0k0
        LOAD_LHS(lt_off0);
      }  // end kcout-1
      // last k unit
      // smr1
      LOAD_SMR_MODE17_BF16_ROW(smr1, rt_off1);
      // m0k0 * smr0
      VMM_MODE17_BF16(0, smr0);
      // m1k0
      LOAD_LHS(lt_off1);
      // m1k0 * smr0
      VMM_MODE17_BF16(32, smr0);

      // m0k1
      LOAD_LHS(lt_off0); // end k new n
      // next n unit smr0
      LOAD_SMR_MODE17_BF16_ROW(smr0, rt_off0);
      __dtu_c_movsr2naccovr(0x1);
      // m0k1 * smr1
      VMM_MODE17_BF16(0, smr1);
      // m1k1
      LOAD_LHS(lt_off2);
      // m1k1 * smr1
      VMM_MODE17_BF16(32, smr1);
      // next n unit m0k0
      LOAD_LHS(lt_off0);
      vab_shift += 1024;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }  // end ncount-1
    __dtu_c_movsr2naccovr(naccovr);
    for (int k = 0; k < K - 64; k += 64) {
      // smr1
      LOAD_SMR_MODE17_BF16_ROW(smr1, rt_off0);
      // m0k0 * smr0
      VMM_MODE17_BF16(0, smr0);
      // m1k0
      LOAD_LHS(lt_off1);
      // m1k0 * smr0
      VMM_MODE17_BF16(32, smr0);

      // m0k1
      LOAD_LHS(lt_off0);
      // next k unit smr0
      LOAD_SMR_MODE17_BF16_ROW(smr0, rt_off0);
      __dtu_c_movsr2naccovr(0x1);
      // m0k1 * smr1
      VMM_MODE17_BF16(0, smr1);
      // m1k1
      LOAD_LHS(lt_off1);
      // m1k1 * smr1
      VMM_MODE17_BF16(32, smr1);

      // next k unit m0k0
      LOAD_LHS(lt_off0);
    }  // end kcout-1
    // last k unit of last n unit
    // smr1
    LOAD_SMR_MODE17_BF16_ROW(smr1, rt_off2);
    // m0k0 * smr0
    VMM_MODE17_BF16(0, smr0);
    // m1k0
    LOAD_LHS(lt_off1);
    // m1k0 * smr0
    VMM_MODE17_BF16(32, smr0);

    // m0k1
    LOAD_LHS(lt_off0);
    // next m unit smr0
    LOAD_SMR_MODE17_BF16_ROW(smr0, rt_off0);
    __dtu_c_movsr2naccovr(0x1);
    // m0k1 * smr1
    VMM_MODE17_BF16(0, smr1);
    // m1k1
    LOAD_LHS(lt_off3);
    // m1k1 * smr1
    VMM_MODE17_BF16(32, smr1);

    // next m unit m0k0
    LOAD_LHS(lt_off0);
    vab_shift += 1024;
    __dtu_c_movsr2vab_m_s1(vab_shift);
    __dtu_c_movsr2vab_m_d(vab_shift);
  }  // end mcount
  // store
    vab_shift = 0;
    __dtu_c_movsr2vab_lv_s(0);
    __dtu_c_movsr2vab_lv_d(0);
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);
    __dtu_c_movsr2vab_m_s2(0);
#pragma clang loop unroll(disable)
    for (int m = 0; m < M; m = m + 64) {
#pragma clang loop unroll(disable)
      for (int n = 0; n < N - 128; n = n + 128) {
        // load(fp32) muti sum reseult C ->qacc[64-127]
        // add qacc[0-63] + qacc[64-127] ->qacc[0-63]
        // load output fp32 - > qacc[64-127]
        LOAD_OUT(bt_off1);
        MOP_ADD_FP32;
        QA2DA;
        STORE_OUT(ot_off1);
        vab_shift += 1024;
        __dtu_c_movsr2vab_lv_s(vab_shift);
        __dtu_c_movsr2vab_lv_d(vab_shift);
        __dtu_c_movsr2vab_m_s1(vab_shift);
        __dtu_c_movsr2vab_m_d(vab_shift);
        __dtu_c_movsr2vab_m_s2(vab_shift);
      }
      LOAD_OUT(bt_off2);
      MOP_ADD_FP32;
      QA2DA;
      STORE_OUT(ot_off2);
      vab_shift += 1024;
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_lv_d(vab_shift);
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
      __dtu_c_movsr2vab_m_s2(vab_shift);
  }
}



#define LOAD_SMR_MODE17_F16_ROW(smr, offset)  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 0);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 1);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 2);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 3);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 4);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 5);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 6);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 7);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 8);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 9);  \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 10); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 11); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 12); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 13); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 14); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 15); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 16); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 17); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 18); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 19); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 20); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 21); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 22); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 23); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 24); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 25); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 26); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 27); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 28); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 29); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, rt_off0, 30); \
  smr = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr, rt_base, offset, 31);

#define VMM_MODE17_F16(pos, smr)  \
  qacc[0+pos] = __dtu_m_vmm2_mode17_f16(qacc[0+pos], vr0, smr);     \
  qacc[1+pos] = __dtu_m_vmm2_mode17_f16(qacc[1+pos], vr1, smr);     \
  qacc[2+pos] = __dtu_m_vmm2_mode17_f16(qacc[2+pos], vr2, smr);     \
  qacc[3+pos] = __dtu_m_vmm2_mode17_f16(qacc[3+pos], vr3, smr);     \
  qacc[4+pos] = __dtu_m_vmm2_mode17_f16(qacc[4+pos], vr4, smr);     \
  qacc[5+pos] = __dtu_m_vmm2_mode17_f16(qacc[5+pos], vr5, smr);     \
  qacc[6+pos] = __dtu_m_vmm2_mode17_f16(qacc[6+pos], vr6, smr);     \
  qacc[7+pos] = __dtu_m_vmm2_mode17_f16(qacc[7+pos], vr7, smr);     \
  qacc[8+pos] = __dtu_m_vmm2_mode17_f16(qacc[8+pos], vr8, smr);     \
  qacc[9+pos] = __dtu_m_vmm2_mode17_f16(qacc[9+pos], vr9, smr);     \
  qacc[10+pos] = __dtu_m_vmm2_mode17_f16(qacc[10+pos], vr10, smr);  \
  qacc[11+pos] = __dtu_m_vmm2_mode17_f16(qacc[11+pos], vr11, smr);  \
  qacc[12+pos] = __dtu_m_vmm2_mode17_f16(qacc[12+pos], vr12, smr);  \
  qacc[13+pos] = __dtu_m_vmm2_mode17_f16(qacc[13+pos], vr13, smr);  \
  qacc[14+pos] = __dtu_m_vmm2_mode17_f16(qacc[14+pos], vr14, smr);  \
  qacc[15+pos] = __dtu_m_vmm2_mode17_f16(qacc[15+pos], vr15, smr);  \
  qacc[16+pos] = __dtu_m_vmm2_mode17_f16(qacc[16+pos], vr16, smr);  \
  qacc[17+pos] = __dtu_m_vmm2_mode17_f16(qacc[17+pos], vr17, smr);  \
  qacc[18+pos] = __dtu_m_vmm2_mode17_f16(qacc[18+pos], vr18, smr);  \
  qacc[19+pos] = __dtu_m_vmm2_mode17_f16(qacc[19+pos], vr19, smr);  \
  qacc[20+pos] = __dtu_m_vmm2_mode17_f16(qacc[20+pos], vr20, smr);  \
  qacc[21+pos] = __dtu_m_vmm2_mode17_f16(qacc[21+pos], vr21, smr);  \
  qacc[22+pos] = __dtu_m_vmm2_mode17_f16(qacc[22+pos], vr22, smr);  \
  qacc[23+pos] = __dtu_m_vmm2_mode17_f16(qacc[23+pos], vr23, smr);  \
  qacc[24+pos] = __dtu_m_vmm2_mode17_f16(qacc[24+pos], vr24, smr);  \
  qacc[25+pos] = __dtu_m_vmm2_mode17_f16(qacc[25+pos], vr25, smr);  \
  qacc[26+pos] = __dtu_m_vmm2_mode17_f16(qacc[26+pos], vr26, smr);  \
  qacc[27+pos] = __dtu_m_vmm2_mode17_f16(qacc[27+pos], vr27, smr);  \
  qacc[28+pos] = __dtu_m_vmm2_mode17_f16(qacc[28+pos], vr28, smr);  \
  qacc[29+pos] = __dtu_m_vmm2_mode17_f16(qacc[29+pos], vr29, smr);  \
  qacc[30+pos] = __dtu_m_vmm2_mode17_f16(qacc[30+pos], vr30, smr);  \
  qacc[31+pos] = __dtu_m_vmm2_mode17_f16(qacc[31+pos], vr31, smr);


__attribute__((device, dtu_maxinum_vacc(1024))) extern "C" void
c_func_hmatmul_general(int a_addr, int b_addr, int c_addr, int M, int N,
  int K) {
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2vab_m_d(0);

  int BPE = 2;
  smr_t smr0, smr1;
  v64i8 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;

  va16f32x4 qacc[128];
  va16f16x2 c_dacc[128];

  auto k_unit = K >> 5;
  auto n_unit = N >> 6;
  auto on_unit = N >> 5;
  // vpt parallel in rhs
  int lt_addr = a_addr >> 6;
  int rt_addr = b_addr >> 7;
  int ot_addr = c_addr >> 7;
  int offset = 0;
  tar_t lt_base = __dtu_c_movsr2targ(TAR_ADDR_WARP(lt_addr, 0));
  offset = TAR_OFF_WARP(k_unit, k_unit);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1 - 63 * k_unit, 1 - 63 * k_unit);  // next k
  tar_t lt_off1 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1 - 64 * k_unit, 1 - 64 * k_unit);  //  new n
  tar_t lt_off2 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1, 1);  // end k end n new m
  tar_t lt_off3 = __dtu_c_movsr2tari(offset, lt_base);

  tar_t rt_base = __dtu_c_movsr2targ((rt_addr) | ((rt_addr) + 1) << 16);
  offset = TAR_OFF_WARP(n_unit, n_unit);
  tar_t rt_off0 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR_OFF_WARP(2 - (K - 1) * n_unit, 2 - (K - 1) * n_unit);
  tar_t rt_off1 = __dtu_c_movsr2tari(offset, rt_base);  // new n
  offset = TAR_OFF_WARP(2 - K * n_unit, 2 - K * n_unit);
  tar_t rt_off2 = __dtu_c_movsr2tari(offset, rt_base);  // new m

  auto bn_unit = N >> 6;
  int bt_addr = (c_addr >> 8) | ((c_addr >> 8) + 1) << 16;
  tar_t bt_base = __dtu_c_movsr2targ(bt_addr);
  offset = (bn_unit << 16) | bn_unit;
  tar_t bt_off0 = __dtu_c_movsr2tari(offset, bt_base);
  offset = (2 - 63 * bn_unit) & 0xffff;
  offset = (offset << 16) | offset;
  tar_t bt_off1 = __dtu_c_movsr2tari(offset, bt_base);
  offset = (2 << 16) | 2;
  tar_t bt_off2 = __dtu_c_movsr2tari(offset, bt_base);

  tar_t ot_base = __dtu_c_movsr2targ((ot_addr) | ((ot_addr) + 2) << 16);
  offset = TAR_OFF_WARP(1, 1);
  tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
  offset = TAR_OFF_WARP(3 - 63 * on_unit, 3 - 63 * on_unit);
  tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);  // new n
  offset = TAR_OFF_WARP(3, 3);
  tar_t ot_off2 = __dtu_c_movsr2tari(offset, ot_base);  // new m
  offset = TAR_OFF_WARP(on_unit - 1, on_unit - 1);
  tar_t ot_off3 = __dtu_c_movsr2tari(offset, ot_base);

  // k0n0
  LOAD_SMR_MODE17_F16_ROW(smr0, rt_off0);
  // m0k0
  LOAD_LHS(lt_off0);
  int naccovr = 0x10001;
  __dtu_c_movsr2naccovr(naccovr);
  __dtu_c_movsr2vab_m_s2(0);
  int vab_shift = 0;
// fp16 vmm2 mode17: [64, 64] * [64, 128] = [64, 128]
#pragma clang loop unroll(full)
  for (int m = 0; m < M; m += 64) {
    for (int n = 0; n < N - 128; n += 128) {  // VPT PARA DIM
      __dtu_c_movsr2naccovr(naccovr);
      for (int k = 0; k < K - 64; k += 64) {
        // smr1
        LOAD_SMR_MODE17_F16_ROW(smr1, rt_off0);
        // m0k0 * smr0
        VMM_MODE17_F16(0, smr0);
        // m1k0
        LOAD_LHS(lt_off1);
        // m1k0 * smr0
        VMM_MODE17_F16(32, smr0);

        // m0k1
        LOAD_LHS(lt_off0);
        // next k unit smr0
        LOAD_SMR_MODE17_F16_ROW(smr0, rt_off0);
        __dtu_c_movsr2naccovr(0x1);
        // m0k1 * smr1
        VMM_MODE17_F16(0, smr1);
        // m1k1
        LOAD_LHS(lt_off1);
        // m1k1 * smr1
        VMM_MODE17_F16(32, smr1);
        // next k unit m0k0
        LOAD_LHS(lt_off0);
      }  // end kcout-1
      // last k unit
      // smr1
      LOAD_SMR_MODE17_F16_ROW(smr1, rt_off1);
      // m0k0 * smr0
      VMM_MODE17_F16(0, smr0);
      // m1k0
      LOAD_LHS(lt_off1);
      // m1k0 * smr0
      VMM_MODE17_F16(32, smr0);

      // m0k1
      LOAD_LHS(lt_off0); // end k new n
      // next n unit smr0
      LOAD_SMR_MODE17_F16_ROW(smr0, rt_off0);
      __dtu_c_movsr2naccovr(0x1);
      // m0k1 * smr1
      VMM_MODE17_F16(0, smr1);
      // m1k1
      LOAD_LHS(lt_off2);
      // m1k1 * smr1
      VMM_MODE17_F16(32, smr1);
      // next n unit m0k0
      LOAD_LHS(lt_off0);
      vab_shift += 1024;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }  // end ncount-1
    __dtu_c_movsr2naccovr(naccovr);
    for (int k = 0; k < K - 64; k += 64) {
      // smr1
      LOAD_SMR_MODE17_F16_ROW(smr1, rt_off0);
      // m0k0 * smr0
      VMM_MODE17_F16(0, smr0);
      // m1k0
      LOAD_LHS(lt_off1);
      // m1k0 * smr0
      VMM_MODE17_F16(32, smr0);

      // m0k1
      LOAD_LHS(lt_off0);
      // next k unit smr0
      LOAD_SMR_MODE17_F16_ROW(smr0, rt_off0);
      __dtu_c_movsr2naccovr(0x1);
      // m0k1 * smr1
      VMM_MODE17_F16(0, smr1);
      // m1k1
      LOAD_LHS(lt_off1);
      // m1k1 * smr1
      VMM_MODE17_F16(32, smr1);

      // next k unit m0k0
      LOAD_LHS(lt_off0);
    }  // end kcout-1
    // last k unit of last n unit
    // smr1
    LOAD_SMR_MODE17_F16_ROW(smr1, rt_off2);
    // m0k0 * smr0
    VMM_MODE17_F16(0, smr0);
    // m1k0
    LOAD_LHS(lt_off1);
    // m1k0 * smr0
    VMM_MODE17_F16(32, smr0);

    // m0k1
    LOAD_LHS(lt_off0);
    // next m unit smr0
    LOAD_SMR_MODE17_F16_ROW(smr0, rt_off0);
    __dtu_c_movsr2naccovr(0x1);
    // m0k1 * smr1
    VMM_MODE17_F16(0, smr1);
    // m1k1
    LOAD_LHS(lt_off3);
    // m1k1 * smr1
    VMM_MODE17_F16(32, smr1);

    // next m unit m0k0
    LOAD_LHS(lt_off0);
    vab_shift += 1024;
    __dtu_c_movsr2vab_m_s1(vab_shift);
    __dtu_c_movsr2vab_m_d(vab_shift);
  }  // end mcount
  // store
    vab_shift = 0;
    __dtu_c_movsr2vab_lv_s(0);
    __dtu_c_movsr2vab_lv_d(0);
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);
    __dtu_c_movsr2vab_m_s2(0);
#pragma clang loop unroll(disable)
    for (int m = 0; m < M; m = m + 64) {
#pragma clang loop unroll(disable)
      for (int n = 0; n < N - 128; n = n + 128) {
        // load(fp32) muti sum reseult C ->qacc[64-127]
        // add qacc[0-63] + qacc[64-127] ->qacc[0-63]
        // load output fp32 - > qacc[64-127]
        LOAD_OUT(bt_off1);
        MOP_ADD_FP32;
        QA2DA;
        STORE_OUT(ot_off1);
        vab_shift += 1024;
        __dtu_c_movsr2vab_lv_s(vab_shift);
        __dtu_c_movsr2vab_lv_d(vab_shift);
        __dtu_c_movsr2vab_m_s1(vab_shift);
        __dtu_c_movsr2vab_m_d(vab_shift);
        __dtu_c_movsr2vab_m_s2(vab_shift);
      }
      LOAD_OUT(bt_off2);
      MOP_ADD_FP32;
      QA2DA;
      STORE_OUT(ot_off2);
      vab_shift += 1024;
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_lv_d(vab_shift);
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
      __dtu_c_movsr2vab_m_s2(vab_shift);
    }
}


#define LOAD_SMR_MODE18_F32_ROW(smr, offset)                                   \
  smr =                                                                        \
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0, 0); \
  smr =                                                                        \
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0, 1); \
  smr =                                                                        \
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0, 2); \
  smr =                                                                        \
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0, 3); \
  smr =                                                                        \
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0, 4); \
  smr =                                                                        \
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0, 5); \
  smr =                                                                        \
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0, 6); \
  smr =                                                                        \
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0, 7); \
  smr =                                                                        \
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0, 8); \
  smr =                                                                        \
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0, 9); \
  smr = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0,   \
                                            10);                               \
  smr = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0,   \
                                            11);                               \
  smr = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0,   \
                                            12);                               \
  smr = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0,   \
                                            13);                               \
  smr = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, rhs_tar_off0,   \
                                            14);                               \
  smr = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr, rhs_tar_base, offset, 15);
#define LOAD_LHS_F32(offset)                                \
  vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);  \
  vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);  \
  vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);  \
  vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);  \
  vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);  \
  vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);  \
  vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);  \
  vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);  \
  vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);  \
  vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);  \
  vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0); \
  vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0); \
  vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0); \
  vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0); \
  vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0); \
  vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, offset);

#define VMM_MODE18_F32(pos, smr)                                               \
  dacc_arr[0 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[0 + pos], vr0, smr);    \
  dacc_arr[1 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[1 + pos], vr1, smr);    \
  dacc_arr[2 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[2 + pos], vr2, smr);    \
  dacc_arr[3 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[3 + pos], vr3, smr);    \
  dacc_arr[4 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[4 + pos], vr4, smr);    \
  dacc_arr[5 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[5 + pos], vr5, smr);    \
  dacc_arr[6 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[6 + pos], vr6, smr);    \
  dacc_arr[7 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[7 + pos], vr7, smr);    \
  dacc_arr[8 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[8 + pos], vr8, smr);    \
  dacc_arr[9 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[9 + pos], vr9, smr);    \
  dacc_arr[10 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[10 + pos], vr10, smr); \
  dacc_arr[11 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[11 + pos], vr11, smr); \
  dacc_arr[12 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[12 + pos], vr12, smr); \
  dacc_arr[13 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[13 + pos], vr13, smr); \
  dacc_arr[14 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[14 + pos], vr14, smr); \
  dacc_arr[15 + pos] = __dtu_m_vmm2_mode18_f32(dacc_arr[15 + pos], vr15, smr);
#define LOAD_OUT_F32(offset)                                            \
  c_dacc_arr[0] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);  \
  c_dacc_arr[1] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);  \
  c_dacc_arr[2] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);  \
  c_dacc_arr[3] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);  \
  c_dacc_arr[4] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);  \
  c_dacc_arr[5] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);  \
  c_dacc_arr[6] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);  \
  c_dacc_arr[7] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);  \
  c_dacc_arr[8] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);  \
  c_dacc_arr[9] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);  \
  c_dacc_arr[10] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[11] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[12] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[13] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[14] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[15] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[16] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[17] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[18] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[19] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[20] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[21] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[22] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[23] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[24] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[25] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[26] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[27] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[28] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[29] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[30] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0); \
  c_dacc_arr[31] = __dtu_l_tvldqa_f32_da(bias_tar_base, offset);

#define STORE_OUT_F32(offset)                                        \
  __dtu_v_tvstda_f32_dual(dacc_arr[0], out_tar_base, out_tar_off0);  \
  __dtu_v_tvstda_f32_dual(dacc_arr[1], out_tar_base, out_tar_off0);  \
  __dtu_v_tvstda_f32_dual(dacc_arr[2], out_tar_base, out_tar_off0);  \
  __dtu_v_tvstda_f32_dual(dacc_arr[3], out_tar_base, out_tar_off0);  \
  __dtu_v_tvstda_f32_dual(dacc_arr[4], out_tar_base, out_tar_off0);  \
  __dtu_v_tvstda_f32_dual(dacc_arr[5], out_tar_base, out_tar_off0);  \
  __dtu_v_tvstda_f32_dual(dacc_arr[6], out_tar_base, out_tar_off0);  \
  __dtu_v_tvstda_f32_dual(dacc_arr[7], out_tar_base, out_tar_off0);  \
  __dtu_v_tvstda_f32_dual(dacc_arr[8], out_tar_base, out_tar_off0);  \
  __dtu_v_tvstda_f32_dual(dacc_arr[9], out_tar_base, out_tar_off0);  \
  __dtu_v_tvstda_f32_dual(dacc_arr[10], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[11], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[12], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[13], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[14], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[15], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[16], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[17], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[18], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[19], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[20], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[21], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[22], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[23], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[24], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[25], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[26], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[27], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[28], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[29], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[30], out_tar_base, out_tar_off0); \
  __dtu_v_tvstda_f32_dual(dacc_arr[31], out_tar_base, offset);

#define MOP_ADD_DACC_FP32                                              \
  dacc_arr[0] = __dtu_m_mop_add_f32_da(dacc_arr[0], c_dacc_arr[0]);    \
  dacc_arr[1] = __dtu_m_mop_add_f32_da(dacc_arr[1], c_dacc_arr[1]);    \
  dacc_arr[2] = __dtu_m_mop_add_f32_da(dacc_arr[2], c_dacc_arr[2]);    \
  dacc_arr[3] = __dtu_m_mop_add_f32_da(dacc_arr[3], c_dacc_arr[3]);    \
  dacc_arr[4] = __dtu_m_mop_add_f32_da(dacc_arr[4], c_dacc_arr[4]);    \
  dacc_arr[5] = __dtu_m_mop_add_f32_da(dacc_arr[5], c_dacc_arr[5]);    \
  dacc_arr[6] = __dtu_m_mop_add_f32_da(dacc_arr[6], c_dacc_arr[6]);    \
  dacc_arr[7] = __dtu_m_mop_add_f32_da(dacc_arr[7], c_dacc_arr[7]);    \
  dacc_arr[8] = __dtu_m_mop_add_f32_da(dacc_arr[8], c_dacc_arr[8]);    \
  dacc_arr[9] = __dtu_m_mop_add_f32_da(dacc_arr[9], c_dacc_arr[9]);    \
  dacc_arr[10] = __dtu_m_mop_add_f32_da(dacc_arr[10], c_dacc_arr[10]); \
  dacc_arr[11] = __dtu_m_mop_add_f32_da(dacc_arr[11], c_dacc_arr[11]); \
  dacc_arr[12] = __dtu_m_mop_add_f32_da(dacc_arr[12], c_dacc_arr[12]); \
  dacc_arr[13] = __dtu_m_mop_add_f32_da(dacc_arr[13], c_dacc_arr[13]); \
  dacc_arr[14] = __dtu_m_mop_add_f32_da(dacc_arr[14], c_dacc_arr[14]); \
  dacc_arr[15] = __dtu_m_mop_add_f32_da(dacc_arr[15], c_dacc_arr[15]); \
  dacc_arr[16] = __dtu_m_mop_add_f32_da(dacc_arr[16], c_dacc_arr[16]); \
  dacc_arr[17] = __dtu_m_mop_add_f32_da(dacc_arr[17], c_dacc_arr[17]); \
  dacc_arr[18] = __dtu_m_mop_add_f32_da(dacc_arr[18], c_dacc_arr[18]); \
  dacc_arr[19] = __dtu_m_mop_add_f32_da(dacc_arr[19], c_dacc_arr[19]); \
  dacc_arr[20] = __dtu_m_mop_add_f32_da(dacc_arr[20], c_dacc_arr[20]); \
  dacc_arr[21] = __dtu_m_mop_add_f32_da(dacc_arr[21], c_dacc_arr[21]); \
  dacc_arr[22] = __dtu_m_mop_add_f32_da(dacc_arr[22], c_dacc_arr[22]); \
  dacc_arr[23] = __dtu_m_mop_add_f32_da(dacc_arr[23], c_dacc_arr[23]); \
  dacc_arr[24] = __dtu_m_mop_add_f32_da(dacc_arr[24], c_dacc_arr[24]); \
  dacc_arr[25] = __dtu_m_mop_add_f32_da(dacc_arr[25], c_dacc_arr[25]); \
  dacc_arr[26] = __dtu_m_mop_add_f32_da(dacc_arr[26], c_dacc_arr[26]); \
  dacc_arr[27] = __dtu_m_mop_add_f32_da(dacc_arr[27], c_dacc_arr[27]); \
  dacc_arr[28] = __dtu_m_mop_add_f32_da(dacc_arr[28], c_dacc_arr[28]); \
  dacc_arr[29] = __dtu_m_mop_add_f32_da(dacc_arr[29], c_dacc_arr[29]); \
  dacc_arr[30] = __dtu_m_mop_add_f32_da(dacc_arr[30], c_dacc_arr[30]); \
  dacc_arr[31] = __dtu_m_mop_add_f32_da(dacc_arr[31], c_dacc_arr[31]);

__attribute__((device, dtu_maxinum_vacc(256))) extern "C" void
c_func_smatmul_general(int a_addr, int b_addr, int c_addr, int M, int N,
  int K) {
  v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2vab_m_d(0);

  va16f32x2 dacc_arr[32];
  va16f32x2 c_dacc_arr[32];

  smr_t smr0;
  smr_t smr1;

  auto k_unit = K >> 4;
  auto n_unit = N >> 5;

  // vpt parallel in rhs
  int lhs_tar_addr = a_addr >> 6;
  int rhs_tar_addr = b_addr >> 7;
  int bias_or_out_tar_addr = c_addr >> 7;

  tar_t lhs_tar_base = __dtu_c_movsr2targ(TAR_ADDR_WARP(lhs_tar_addr, 0));
  int lhs_off0 = TAR_OFF_WARP(k_unit, k_unit);
  tar_t lhs_tar_off0 = __dtu_c_movsr2tari(lhs_off0, lhs_tar_base);
  int lhs_off1 = TAR_OFF_WARP(1 - 31 * k_unit, 1 - 31 * k_unit);  // next k
  tar_t lhs_tar_off1 = __dtu_c_movsr2tari(lhs_off1, lhs_tar_base);
  int lhs_off2 =
      TAR_OFF_WARP(1 - 32 * k_unit, 1 - 32 * k_unit);  //  end k new n
  tar_t lhs_tar_off2 = __dtu_c_movsr2tari(lhs_off2, lhs_tar_base);
  int lhs_off3 = TAR_OFF_WARP(1, 1);  // end k end n new m
  tar_t lhs_tar_off3 = __dtu_c_movsr2tari(lhs_off3, lhs_tar_base);

  tar_t rhs_tar_base =
      __dtu_c_movsr2targ((rhs_tar_addr) | ((rhs_tar_addr) + 1) << 16);
  int rhs_off0 = TAR_OFF_WARP(n_unit, n_unit);
  tar_t rhs_tar_off0 = __dtu_c_movsr2tari(rhs_off0, rhs_tar_base);
  int rhs_off1 = TAR_OFF_WARP(2 + n_unit - K * n_unit, 2 + n_unit - K * n_unit);
  tar_t rhs_tar_off1 = __dtu_c_movsr2tari(rhs_off1, rhs_tar_base);
  int rhs_off2 = TAR_OFF_WARP(2 - K * n_unit, 2 - K * n_unit);
  tar_t rhs_tar_off2 = __dtu_c_movsr2tari(rhs_off2, rhs_tar_base);

  tar_t bias_tar_base = __dtu_c_movsr2targ((bias_or_out_tar_addr) |
                                           ((bias_or_out_tar_addr) + 1) << 16);
  int bias_off0 = TAR_OFF_WARP(n_unit, n_unit);
  tar_t bias_tar_off0 = __dtu_c_movsr2tari(bias_off0, bias_tar_base);
  int bias_off1 = TAR_OFF_WARP(2 - 31 * n_unit, 2 - 31 * n_unit);
  tar_t bias_tar_off1 = __dtu_c_movsr2tari(bias_off1, bias_tar_base);
  int bias_off2 = TAR_OFF_WARP(2, 2);
  tar_t bias_tar_off2 = __dtu_c_movsr2tari(bias_off2, bias_tar_base);

  tar_t out_tar_base = __dtu_c_movsr2targ((bias_or_out_tar_addr) |
                                          ((bias_or_out_tar_addr) + 1) << 16);
  int out_off0 = TAR_OFF_WARP(n_unit, n_unit);
  tar_t out_tar_off0 = __dtu_c_movsr2tari(out_off0, out_tar_base);
  int out_off1 = TAR_OFF_WARP(2 - 31 * n_unit, 2 - 31 * n_unit);
  tar_t out_tar_off1 = __dtu_c_movsr2tari(out_off1, out_tar_base);
  int out_off2 = TAR_OFF_WARP(2, 2);
  tar_t out_tar_off2 = __dtu_c_movsr2tari(out_off2, out_tar_base);

  LOAD_SMR_MODE18_F32_ROW(smr0, rhs_tar_off0);
  LOAD_LHS_F32(lhs_tar_off0);

  int naccovr = 0x10001;
  __dtu_c_movsr2naccovr(naccovr);
  __dtu_c_movsr2vab_m_s2(0);
  int vab_shift = 0;
// [32, 32] * [32, 64] = [32, 64]
#pragma clang loop unroll(full)
  for (int m = 0; m < M; m += 32) {
    for (int n = 0; n < N - 64; n += 64) {  // VPT PARA DIM
      __dtu_c_movsr2naccovr(naccovr);
      for (int k = 0; k < K - 32; k += 32) {
        // m0k0
        VMM_MODE18_F32(0, smr0);
        LOAD_LHS_F32(lhs_tar_off1);
        LOAD_SMR_MODE18_F32_ROW(smr1, rhs_tar_off0);
        // m1k0
        VMM_MODE18_F32(16, smr0);
        LOAD_LHS_F32(lhs_tar_off0);
        __dtu_c_movsr2naccovr(0x1);

        // m0k1
        VMM_MODE18_F32(0, smr1);
        LOAD_LHS_F32(lhs_tar_off1);
        LOAD_SMR_MODE18_F32_ROW(smr0, rhs_tar_off0);
        // m1k1
        VMM_MODE18_F32(16, smr1);
        LOAD_LHS_F32(lhs_tar_off0);
      }  // end kcout-1

      // m0k0
      VMM_MODE18_F32(0, smr0);
      LOAD_LHS_F32(lhs_tar_off1);
      LOAD_SMR_MODE18_F32_ROW(smr1, rhs_tar_off1);

      // m1k0
      VMM_MODE18_F32(16, smr0);
      LOAD_LHS_F32(lhs_tar_off0);
      __dtu_c_movsr2naccovr(0x1);

      // m0k1
      VMM_MODE18_F32(0, smr1);
      LOAD_LHS_F32(lhs_tar_off2);
      LOAD_SMR_MODE18_F32_ROW(smr0, rhs_tar_off0);

      // m1k1
      VMM_MODE18_F32(16, smr1);
      LOAD_LHS_F32(lhs_tar_off0);
      vab_shift += 256;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }  // end ncount-1
    __dtu_c_movsr2naccovr(naccovr);
    for (int k = 0; k < K - 32; k += 32) {
      // m0k0
      VMM_MODE18_F32(0, smr0);
      LOAD_LHS_F32(lhs_tar_off1);
      LOAD_SMR_MODE18_F32_ROW(smr1, rhs_tar_off0);

      // m1k0
      VMM_MODE18_F32(16, smr0);
      LOAD_LHS_F32(lhs_tar_off0);
      __dtu_c_movsr2naccovr(0x1);

      // m0k1
      VMM_MODE18_F32(0, smr1);
      LOAD_LHS_F32(lhs_tar_off1);
      LOAD_SMR_MODE18_F32_ROW(smr0, rhs_tar_off0);

      // m1k1
      VMM_MODE18_F32(16, smr1);
      LOAD_LHS_F32(lhs_tar_off0);
      __dtu_c_movsr2naccovr(0x1);
    }  // end kcout-1

    // m0k0
    VMM_MODE18_F32(0, smr0);
    LOAD_LHS_F32(lhs_tar_off1);
    LOAD_SMR_MODE18_F32_ROW(smr1, rhs_tar_off2);

    // m1k0
    VMM_MODE18_F32(16, smr0);
    LOAD_LHS_F32(lhs_tar_off0);
    __dtu_c_movsr2naccovr(0x1);

    // m0k1
    VMM_MODE18_F32(0, smr1);
    LOAD_LHS_F32(lhs_tar_off3);
    LOAD_SMR_MODE18_F32_ROW(smr0, rhs_tar_off0);

    // m1k1
    VMM_MODE18_F32(16, smr1);
    LOAD_LHS_F32(lhs_tar_off0);
    vab_shift += 256;
    __dtu_c_movsr2vab_m_s1(vab_shift);
    __dtu_c_movsr2vab_m_d(vab_shift);
  }  // end mcount
  // store
    vab_shift = 0;
    __dtu_c_movsr2vab_lv_s(0);
    __dtu_c_movsr2vab_lv_d(0);
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);
    __dtu_c_movsr2vab_m_s2(0);

#pragma clang loop unroll(disable)
    for (int m = 0; m < M; m = m + 32) {
#pragma clang loop unroll(disable)
      for (int n = 0; n < N - 64; n = n + 64) {
        LOAD_OUT_F32(bias_tar_off1);
        MOP_ADD_DACC_FP32;
        STORE_OUT_F32(out_tar_off1);

        vab_shift += 256;
        __dtu_c_movsr2vab_lv_s(vab_shift);
        __dtu_c_movsr2vab_lv_d(vab_shift);
        __dtu_c_movsr2vab_m_s1(vab_shift);
        __dtu_c_movsr2vab_m_d(vab_shift);
        __dtu_c_movsr2vab_m_s2(vab_shift);
      }
      LOAD_OUT_F32(bias_tar_off2);
      MOP_ADD_DACC_FP32;
      STORE_OUT_F32(out_tar_off2);

      vab_shift += 256;
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_lv_d(vab_shift);
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
      __dtu_c_movsr2vab_m_s2(vab_shift);
    }
}


#define LOAD_SMR_MODE19_S8_ROW(smr, offset)                            \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 0);  \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 1);  \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 2);  \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 3);  \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 4);  \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 5);  \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 6);  \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 7);  \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 8);  \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 9);  \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 10); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 11); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 12); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 13); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 14); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 15); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 16); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 17); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 18); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 19); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 20); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 21); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 22); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 23); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 24); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 25); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 26); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 27); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 28); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 29); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, rt_off0, 30); \
  smr = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr, rt_base, offset, 31);
#define LOAD_LHS_S8(offset)                   \
  vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);  \
  vr10 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr11 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr12 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr13 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr14 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr15 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr16 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr17 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr18 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr19 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr20 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr21 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr22 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr23 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr24 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr25 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr26 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr27 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr28 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr29 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr30 = __dtu_s_tvld_itar(lt_base, lt_off0); \
  vr31 = __dtu_s_tvld_itar(lt_base, offset);
#define VMM_MODE19_S8(pos, smr)                                            \
  qacc[0 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[0 + pos], vr0, smr);    \
  qacc[1 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[1 + pos], vr1, smr);    \
  qacc[2 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[2 + pos], vr2, smr);    \
  qacc[3 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[3 + pos], vr3, smr);    \
  qacc[4 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[4 + pos], vr4, smr);    \
  qacc[5 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[5] + pos, vr5, smr);    \
  qacc[6 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[6 + pos], vr6, smr);    \
  qacc[7 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[7 + pos], vr7, smr);    \
  qacc[8 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[8 + pos], vr8, smr);    \
  qacc[9 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[9 + pos], vr9, smr);    \
  qacc[10 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[10 + pos], vr10, smr); \
  qacc[11 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[11 + pos], vr11, smr); \
  qacc[12 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[12 + pos], vr12, smr); \
  qacc[13 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[13 + pos], vr13, smr); \
  qacc[14 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[14 + pos], vr14, smr); \
  qacc[15 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[15 + pos], vr15, smr); \
  qacc[16 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[16 + pos], vr16, smr); \
  qacc[17 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[17 + pos], vr17, smr); \
  qacc[18 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[18 + pos], vr18, smr); \
  qacc[19 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[19 + pos], vr19, smr); \
  qacc[20 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[20 + pos], vr20, smr); \
  qacc[21 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[21 + pos], vr21, smr); \
  qacc[22 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[22 + pos], vr22, smr); \
  qacc[23 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[23 + pos], vr23, smr); \
  qacc[24 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[24 + pos], vr24, smr); \
  qacc[25 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[25 + pos], vr25, smr); \
  qacc[26 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[26 + pos], vr26, smr); \
  qacc[27 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[27 + pos], vr27, smr); \
  qacc[28 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[28 + pos], vr28, smr); \
  qacc[29 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[29 + pos], vr29, smr); \
  qacc[30 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[30 + pos], vr30, smr); \
  qacc[31 + pos] = __dtu_m_vmm2_mode19_s8_acc0(qacc[31 + pos], vr31, smr);

#define LOAD_OUT_S32(offset)                            \
  c_qacc[0] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0);  \
  c_qacc[1] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0);  \
  c_qacc[2] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0);  \
  c_qacc[3] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0);  \
  c_qacc[4] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0);  \
  c_qacc[5] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0);  \
  c_qacc[6] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0);  \
  c_qacc[7] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0);  \
  c_qacc[8] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0);  \
  c_qacc[9] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0);  \
  c_qacc[10] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[11] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[12] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[13] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[14] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[15] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[16] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[17] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[18] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[19] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[20] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[21] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[22] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[23] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[24] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[25] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[26] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[27] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[28] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[29] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[30] = __dtu_l_tvldqa_s32_qa(bt_base, bt_off0); \
  c_qacc[31] = __dtu_l_tvldqa_s32_qa(bt_base, offset);

#define STORE_OUT_S32(offset)                          \
  __dtu_v_tvstda_s32_dual(dacc[0], ot_base, ot_off0);  \
  __dtu_v_tvstda_s32_dual(dacc[1], ot_base, ot_off3);  \
  __dtu_v_tvstda_s32_dual(dacc[2], ot_base, ot_off0);  \
  __dtu_v_tvstda_s32_dual(dacc[3], ot_base, ot_off3);  \
  __dtu_v_tvstda_s32_dual(dacc[4], ot_base, ot_off0);  \
  __dtu_v_tvstda_s32_dual(dacc[5], ot_base, ot_off3);  \
  __dtu_v_tvstda_s32_dual(dacc[6], ot_base, ot_off0);  \
  __dtu_v_tvstda_s32_dual(dacc[7], ot_base, ot_off3);  \
  __dtu_v_tvstda_s32_dual(dacc[8], ot_base, ot_off0);  \
  __dtu_v_tvstda_s32_dual(dacc[9], ot_base, ot_off3);  \
  __dtu_v_tvstda_s32_dual(dacc[10], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[11], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[12], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[13], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[14], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[15], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[16], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[17], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[18], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[19], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[20], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[21], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[22], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[23], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[24], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[25], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[26], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[27], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[28], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[29], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[30], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[31], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[32], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[33], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[34], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[35], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[36], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[37], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[38], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[39], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[40], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[41], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[42], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[43], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[44], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[45], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[46], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[47], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[48], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[49], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[50], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[51], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[52], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[53], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[54], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[55], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[56], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[57], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[58], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[59], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[60], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[61], ot_base, ot_off3); \
  __dtu_v_tvstda_s32_dual(dacc[62], ot_base, ot_off0); \
  __dtu_v_tvstda_s32_dual(dacc[63], ot_base, offset);

#define MOP_ADD_QACC_S32                                   \
  qacc[0] = __dtu_m_mop_add_s32_qa(qacc[0], c_qacc[0]);    \
  qacc[1] = __dtu_m_mop_add_s32_qa(qacc[1], c_qacc[1]);    \
  qacc[2] = __dtu_m_mop_add_s32_qa(qacc[2], c_qacc[2]);    \
  qacc[3] = __dtu_m_mop_add_s32_qa(qacc[3], c_qacc[3]);    \
  qacc[4] = __dtu_m_mop_add_s32_qa(qacc[4], c_qacc[4]);    \
  qacc[5] = __dtu_m_mop_add_s32_qa(qacc[5], c_qacc[5]);    \
  qacc[6] = __dtu_m_mop_add_s32_qa(qacc[6], c_qacc[6]);    \
  qacc[7] = __dtu_m_mop_add_s32_qa(qacc[7], c_qacc[7]);    \
  qacc[8] = __dtu_m_mop_add_s32_qa(qacc[8], c_qacc[8]);    \
  qacc[9] = __dtu_m_mop_add_s32_qa(qacc[9], c_qacc[9]);    \
  qacc[10] = __dtu_m_mop_add_s32_qa(qacc[10], c_qacc[10]); \
  qacc[11] = __dtu_m_mop_add_s32_qa(qacc[11], c_qacc[11]); \
  qacc[12] = __dtu_m_mop_add_s32_qa(qacc[12], c_qacc[12]); \
  qacc[13] = __dtu_m_mop_add_s32_qa(qacc[13], c_qacc[13]); \
  qacc[14] = __dtu_m_mop_add_s32_qa(qacc[14], c_qacc[14]); \
  qacc[15] = __dtu_m_mop_add_s32_qa(qacc[15], c_qacc[15]); \
  qacc[16] = __dtu_m_mop_add_s32_qa(qacc[16], c_qacc[16]); \
  qacc[17] = __dtu_m_mop_add_s32_qa(qacc[17], c_qacc[17]); \
  qacc[18] = __dtu_m_mop_add_s32_qa(qacc[18], c_qacc[18]); \
  qacc[19] = __dtu_m_mop_add_s32_qa(qacc[19], c_qacc[19]); \
  qacc[20] = __dtu_m_mop_add_s32_qa(qacc[20], c_qacc[20]); \
  qacc[21] = __dtu_m_mop_add_s32_qa(qacc[21], c_qacc[21]); \
  qacc[22] = __dtu_m_mop_add_s32_qa(qacc[22], c_qacc[22]); \
  qacc[23] = __dtu_m_mop_add_s32_qa(qacc[23], c_qacc[23]); \
  qacc[24] = __dtu_m_mop_add_s32_qa(qacc[24], c_qacc[24]); \
  qacc[25] = __dtu_m_mop_add_s32_qa(qacc[25], c_qacc[25]); \
  qacc[26] = __dtu_m_mop_add_s32_qa(qacc[26], c_qacc[26]); \
  qacc[27] = __dtu_m_mop_add_s32_qa(qacc[27], c_qacc[27]); \
  qacc[28] = __dtu_m_mop_add_s32_qa(qacc[28], c_qacc[28]); \
  qacc[29] = __dtu_m_mop_add_s32_qa(qacc[29], c_qacc[29]); \
  qacc[30] = __dtu_m_mop_add_s32_qa(qacc[30], c_qacc[30]); \
  qacc[31] = __dtu_m_mop_add_s32_qa(qacc[31], c_qacc[31]);

#define QA2DA_S32                             \
  dacc[0] = __dtu_extractqa2da(qacc[0], 0);   \
  dacc[1] = __dtu_extractqa2da(qacc[0], 1);   \
  dacc[2] = __dtu_extractqa2da(qacc[1], 0);   \
  dacc[3] = __dtu_extractqa2da(qacc[1], 1);   \
  dacc[4] = __dtu_extractqa2da(qacc[2], 0);   \
  dacc[5] = __dtu_extractqa2da(qacc[2], 1);   \
  dacc[6] = __dtu_extractqa2da(qacc[3], 0);   \
  dacc[7] = __dtu_extractqa2da(qacc[3], 1);   \
  dacc[8] = __dtu_extractqa2da(qacc[4], 0);   \
  dacc[9] = __dtu_extractqa2da(qacc[4], 1);   \
  dacc[10] = __dtu_extractqa2da(qacc[5], 0);  \
  dacc[11] = __dtu_extractqa2da(qacc[5], 1);  \
  dacc[12] = __dtu_extractqa2da(qacc[6], 0);  \
  dacc[13] = __dtu_extractqa2da(qacc[6], 1);  \
  dacc[14] = __dtu_extractqa2da(qacc[7], 0);  \
  dacc[15] = __dtu_extractqa2da(qacc[7], 1);  \
  dacc[16] = __dtu_extractqa2da(qacc[8], 0);  \
  dacc[17] = __dtu_extractqa2da(qacc[8], 1);  \
  dacc[18] = __dtu_extractqa2da(qacc[9], 0);  \
  dacc[19] = __dtu_extractqa2da(qacc[9], 1);  \
  dacc[20] = __dtu_extractqa2da(qacc[10], 0); \
  dacc[21] = __dtu_extractqa2da(qacc[10], 1); \
  dacc[22] = __dtu_extractqa2da(qacc[11], 0); \
  dacc[23] = __dtu_extractqa2da(qacc[11], 1); \
  dacc[24] = __dtu_extractqa2da(qacc[12], 0); \
  dacc[25] = __dtu_extractqa2da(qacc[12], 1); \
  dacc[26] = __dtu_extractqa2da(qacc[13], 0); \
  dacc[27] = __dtu_extractqa2da(qacc[13], 1); \
  dacc[28] = __dtu_extractqa2da(qacc[14], 0); \
  dacc[29] = __dtu_extractqa2da(qacc[14], 1); \
  dacc[30] = __dtu_extractqa2da(qacc[15], 0); \
  dacc[31] = __dtu_extractqa2da(qacc[15], 1); \
  dacc[32] = __dtu_extractqa2da(qacc[16], 0); \
  dacc[33] = __dtu_extractqa2da(qacc[16], 1); \
  dacc[34] = __dtu_extractqa2da(qacc[17], 0); \
  dacc[35] = __dtu_extractqa2da(qacc[17], 1); \
  dacc[36] = __dtu_extractqa2da(qacc[18], 0); \
  dacc[37] = __dtu_extractqa2da(qacc[18], 1); \
  dacc[38] = __dtu_extractqa2da(qacc[19], 0); \
  dacc[39] = __dtu_extractqa2da(qacc[19], 1); \
  dacc[40] = __dtu_extractqa2da(qacc[20], 0); \
  dacc[41] = __dtu_extractqa2da(qacc[20], 1); \
  dacc[42] = __dtu_extractqa2da(qacc[21], 0); \
  dacc[43] = __dtu_extractqa2da(qacc[21], 1); \
  dacc[44] = __dtu_extractqa2da(qacc[22], 0); \
  dacc[45] = __dtu_extractqa2da(qacc[22], 1); \
  dacc[46] = __dtu_extractqa2da(qacc[23], 0); \
  dacc[47] = __dtu_extractqa2da(qacc[23], 1); \
  dacc[48] = __dtu_extractqa2da(qacc[24], 0); \
  dacc[49] = __dtu_extractqa2da(qacc[24], 1); \
  dacc[50] = __dtu_extractqa2da(qacc[25], 0); \
  dacc[51] = __dtu_extractqa2da(qacc[25], 1); \
  dacc[52] = __dtu_extractqa2da(qacc[26], 0); \
  dacc[53] = __dtu_extractqa2da(qacc[26], 1); \
  dacc[54] = __dtu_extractqa2da(qacc[27], 0); \
  dacc[55] = __dtu_extractqa2da(qacc[27], 1); \
  dacc[56] = __dtu_extractqa2da(qacc[28], 0); \
  dacc[57] = __dtu_extractqa2da(qacc[28], 1); \
  dacc[58] = __dtu_extractqa2da(qacc[29], 0); \
  dacc[59] = __dtu_extractqa2da(qacc[29], 1); \
  dacc[60] = __dtu_extractqa2da(qacc[30], 0); \
  dacc[61] = __dtu_extractqa2da(qacc[30], 1); \
  dacc[62] = __dtu_extractqa2da(qacc[31], 0); \
  dacc[63] = __dtu_extractqa2da(qacc[31], 1);

__attribute__((device, dtu_maxinum_vacc(512))) extern "C" void
c_func_matmul_general_int8(int a_addr, int b_addr, int c_addr, int M, int N,
                           int K) {
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2vab_m_d(0);
  smr_t smr0, smr1;
  v64i8 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;

  va16i32x4 qacc[32];
  va16i32x4 c_qacc[32];
  va16f16x2 dacc[64];

  auto k_unit = K >> 6;
  auto n_unit = N >> 6;
  auto n2_unit = N >> 7;
  auto on_unit = N >> 5;

  // vpt parallel in rhs
  int lt_addr = a_addr >> 6;
  int rt_addr = b_addr >> 6;
  int ot_addr = c_addr >> 7;
  int offset = 0;
  tar_t lt_base = __dtu_c_movsr2targ(TAR_ADDR_WARP(lt_addr, 0));
  offset = TAR_OFF_WARP(k_unit, k_unit);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1 - 31 * k_unit, 1 - 31 * k_unit);  // next k
  tar_t lt_off1 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1 - 32 * k_unit, 1 - 32 * k_unit);  //  new n
  tar_t lt_off2 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1, 1);  // end k end n new m
  tar_t lt_off3 = __dtu_c_movsr2tari(offset, lt_base);

  tar_t rt_base = __dtu_c_movsr2targ((rt_addr) | ((rt_addr) + 1) << 16);
  offset = TAR_OFF_WARP(n_unit, n_unit);  // 2 row
  tar_t rt_off0 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR_OFF_WARP(1 - (K >> 1) * n_unit, 1 - (K >> 1) * n_unit);
  tar_t rt_off1 = __dtu_c_movsr2tari(offset, rt_base);  // new n
  offset = TAR_OFF_WARP(1 + n2_unit - ((K >> 1) + 1) * n_unit,
                        1 + n2_unit - ((K >> 1) + 1) * n_unit);
  tar_t rt_off2 = __dtu_c_movsr2tari(offset, rt_base);  // new m

  tar_t ot_base = __dtu_c_movsr2targ((ot_addr) | ((ot_addr) + 2) << 16);
  offset = TAR_OFF_WARP(1, 1);
  tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
  offset = TAR_OFF_WARP(3 - 31 * on_unit, 3 - 31 * on_unit);
  tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);  // new n
  offset = TAR_OFF_WARP(3, 3);
  tar_t ot_off2 = __dtu_c_movsr2tari(offset, ot_base);  // new m
  offset = TAR_OFF_WARP(on_unit - 1, on_unit - 1);
  tar_t ot_off3 = __dtu_c_movsr2tari(offset, ot_base);

  auto bt_unit = N >> 6;
  tar_t bt_base = __dtu_c_movsr2targ((c_addr >> 8) | ((c_addr >> 8) + 1) << 16);
  offset = TAR_OFF_WARP(bt_unit, bt_unit);
  tar_t bt_off0 = __dtu_c_movsr2tari(offset, bt_base);
  offset = TAR_OFF_WARP(2 - 31 * bt_unit, 2 - 31 * bt_unit);
  tar_t bt_off1 = __dtu_c_movsr2tari(offset, bt_base);  // new n
  offset = TAR_OFF_WARP(2, 2);
  tar_t bt_off2 = __dtu_c_movsr2tari(offset, bt_base);  // new m

  LOAD_SMR_MODE19_S8_ROW(smr0, rt_off0);
  // m0k0
  LOAD_LHS_S8(lt_off1);

  int naccovr = 0x10001;
  __dtu_c_movsr2naccovr(naccovr);
  __dtu_c_movsr2vab_m_s2(0);
  int vab_shift = 0;
// fp16 vmm2 mode17: [32, 64] * [64, 128] = [64, 128]
#pragma clang loop unroll(full)
  for (int m = 0; m < M; m += 32) {
    for (int n = 0; n < N - 128; n += 128) {  // VPT PARA DIM
      __dtu_c_movsr2naccovr(naccovr);
      for (int k = 0; k < K - 128; k += 128) {
        // m0k0 * smr0
        VMM_MODE19_S8(0, smr0);
        // m0k1
        LOAD_LHS_S8(lt_off1);
        // smr1
        LOAD_SMR_MODE19_S8_ROW(smr1, rt_off0);
        __dtu_c_movsr2naccovr(0x1);
        // m0k1 * smr1
        VMM_MODE19_S8(0, smr1);
        // next k unit smr0
        LOAD_SMR_MODE19_S8_ROW(smr0, rt_off0);
        // next k unit m0k0
        LOAD_LHS_S8(lt_off1);
      }  // end kcout-1
      // last k unit
      // m0k0 * smr0
      VMM_MODE19_S8(0, smr0);
      // m0k1
      LOAD_LHS_S8(lt_off2);  // end k new n
      // smr1
      LOAD_SMR_MODE19_S8_ROW(smr1, rt_off0);
      __dtu_c_movsr2naccovr(0x1);
      // m0k1 * smr1
      VMM_MODE19_S8(0, smr1);
      // move to n unit smr0
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off1, 0);
      // next n unit smr0
      LOAD_SMR_MODE19_S8_ROW(smr0, rt_off0);
      LOAD_LHS_S8(lt_off1);

      vab_shift += 512;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }  // end ncount-1
    __dtu_c_movsr2naccovr(naccovr);
    for (int k = 0; k < K - 128; k += 128) {
      // m0k0 * smr0
      VMM_MODE19_S8(0, smr0);
      // m0k1
      LOAD_LHS_S8(lt_off1);
      // smr1
      LOAD_SMR_MODE19_S8_ROW(smr1, rt_off0);
      __dtu_c_movsr2naccovr(0x1);
      // m0k1 * smr1
      VMM_MODE19_S8(0, smr1);
      // next k unit smr0
      LOAD_SMR_MODE19_S8_ROW(smr0, rt_off0);
      // next k unit m0k0
      LOAD_LHS_S8(lt_off1);
    }  // end kcout-1
    // last k unit of last n unit m0k0 * smr0
    VMM_MODE19_S8(0, smr0);
    // m0k1
    LOAD_LHS_S8(lt_off3);  // end k  end n new m
    // smr1
    LOAD_SMR_MODE19_S8_ROW(smr1, rt_off0);
    __dtu_c_movsr2naccovr(0x1);
    // m0k1 * smr1
    VMM_MODE19_S8(0, smr1);
    // back to begin of [k, n]
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off2, 0);
    // next m unit smr0
    LOAD_SMR_MODE19_S8_ROW(smr0, rt_off0);
    // next n unit m0k0
    LOAD_LHS_S8(lt_off1);
    vab_shift += 512;
    __dtu_c_movsr2vab_m_s1(vab_shift);
    __dtu_c_movsr2vab_m_d(vab_shift);
  }  // end mcount
     // store
  vab_shift = 0;
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);
  __dtu_c_movsr2vab_m_s2(0);
#pragma clang loop unroll(disable)
  for (int m = 0; m < M; m = m + 32) {
#pragma clang loop unroll(disable)
    for (int n = 0; n < N - 128; n = n + 128) {
      LOAD_OUT_S32(bt_off1);
      MOP_ADD_QACC_S32;
      QA2DA_S32;
      STORE_OUT_S32(ot_off1);
      vab_shift += 512;
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_lv_d(vab_shift);
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
      __dtu_c_movsr2vab_m_s2(vab_shift);
    }
    LOAD_OUT_S32(bt_off2);
    MOP_ADD_QACC_S32;
    QA2DA_S32;
    STORE_OUT_S32(ot_off2);
    vab_shift += 512;
    __dtu_c_movsr2vab_lv_s(vab_shift);
    __dtu_c_movsr2vab_lv_d(vab_shift);
    __dtu_c_movsr2vab_m_s1(vab_shift);
    __dtu_c_movsr2vab_m_d(vab_shift);
    __dtu_c_movsr2vab_m_s2(vab_shift);
  }
}
