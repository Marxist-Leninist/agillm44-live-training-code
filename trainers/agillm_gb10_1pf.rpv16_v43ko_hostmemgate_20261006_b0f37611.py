# V43KN_HOSTMEM 2026-10-06 feell-rpv16-lane (for Scott): v43km (c1feaf1e) + host-memory hygiene, execution-only:
#   gc.collect()+malloc_trim(0) after each successful checkpoint and every AGILLM_MALLOC_TRIM_EVERY_UPDATES (256)
#   optimizer updates; event host_mem_trim. Why: trainer RssAnon grew 22->54.7 GB over 44 h in a 64.76 GB cgroup, every
#   save fell back to sync (low_host_memory, 23-124 s stalls) and the 2026-10-06 03:43Z signal save was OOM-killed.
#   AGILLM_MALLOC_TRIM=0 disables (then identical to v43km).
# V43KM_CSLT_DOWN 2026-10-04 cowork-fable: v43jm (becff735, exact FP32-master persistence) + cuSPARSELt 2:4 expert
#   down projection on sm_86 (block _install_cslt_down_runtime; AGILLM_CSLT_DOWN=0 disables).
# V43J_SPEED 2026-10-04 cowork-fable: v43i + exact execution-only speedups (FSX/WQD/AMX/SFQ, see block).
# V91P5_EMU_INT8 2026-09-28: exact INT8 tensor-core emulation of the sparse NVFP4 GEMM (auto on GeForce sm_86/sm_89).
# V91P4_EMU_EPILOGUE 2026-09-28: emulated sparse GEMM epilogue fused into one Triton pass (bit-identical to V91P3).
# V91P3_PORTABLE 2026-09-28: + portable sparse NVFP4 backend (emulated GEMM, identical quantizer/mask/checkpoint)
#   so the 2B runs on non-Blackwell GPUs (RTX 3090 etc.); on SM12x the native kernel is used as before.
# V91P2_LAZYWQ 2026-09-27: + exact execution-only memory fix (SparseLinear._wq built on demand in backward and
#   released after use instead of 288 resident BF16 copies, ~3.8 GB) so the 2B fits a 24 GB RTX PRO 4000 (sm_120).
# V91P_SM120 2026-09-27: v91 portable (sha 1768b9d9) + SM12x device gate only, for the RTX 5090 (sm_120) replacement host.
#   Runtime .so files rebuilt for sm_120a/x86_64 (GB10 binaries are sm_121a/aarch64). No model, optimizer, objective,
#   data, schedule or checkpoint-schema change. Resume lineage: HF recovery/rpv16-targetfix-step000246907-v6.ckpt.
# V43_CKPTSAFE 2026-09-25: base v42_prod_multimode_guard(d7ab9a5f) + time-save + resume-newest + guarded delete-then-retry (25G floor) + ckpt_save_status.json
# V43C 2026-09-25: fix _checkpoint double reason (v43b TypeError on first save); never-delete Standard refuges
# V42_TIMESAVE 2026-09-25: v41 + wall-clock checkpoint every AGILLM_CKPT_EVERY_SEC (default 1200s) in addition to the update interval.
# V41_CKPT_SURVIVE 2026-09-25: checkpoint I/O failure is non-fatal; last-good checkpoint remains authoritative.
# COMPOSER_V37_FLOOR01 utc=2026-09-24T13:39:05Z ALLHEADS_PROB_FLOOR 0.05->0.01 so AR<=0.98 can bind
from __future__ import annotations

# BEGIN AGILLM_2B_TRAIN_INFER_20261006
# Standalone CLI additions. The original training program below is unchanged.
# Runtime sources are embedded and hash-verified; weights and datasets remain separate.
import sys as _ag_sys
import os as _ag_os
import json as _ag_json
import hashlib as _ag_hash
import pathlib as _ag_path
import base64 as _ag_b64
import io as _ag_io
import zipfile as _ag_zip
import tempfile as _ag_temp

_AGILLM_ORIGINAL_TRAINER_SHA256 = "e50e3a17e37ecdb0bed3607b9299e31c5a5d570631bf2c179e891abdd554cb96"
_AGILLM_RUNTIME_SHA256 = "20d3f6f59959845393ac7fde3254b2b1bdfe0f1a2db0997f4030c08458c4d2e5"
_AGILLM_RUNTIME_B85 = "P)h>@6aWAK2mk;8MqRp;5C>oj005RB000#L003ieZgypIbYEj=Wn*h_Z)t9HE^v9BS!r+EMjHOEUomCBkgY@?X}t@jTENJr+X8Kpt({^4uR)B+p)?V%%#ey*umAf#@67O&?4%6>OLM&UJ^K2qlSZqPwJc6Vv1iqNy)BCuvs9Hii=wot8zrKM$-F96&3I9iHLqn^=-I5@Q=5t_E!y=aZoh50-ez*$E<fn9u!t*O(?N^%9hM!sx>PY%SJ#*SdVR%K?CfIp%g=9KU;Y+dz5nsYpZ{Yv{TahQFXVMygdf9CLT<LTH*a@#2Rr+NowD0;xWrv0RINj71hcE3E`NLd&*<OpFW+2)-h(HLs8Z#J0h+R@r&}qH!G5$AJZX0yXR}!%Ql_{3{QHY2m6`BK{H2k4&f;xT?3PRxwUL#5!_H36zjGe}Tl|CRS}hHWTikFe4J|r9FbAJxy)`-w%Ssf!r`BG;xMrz^Ggr1C%&t`<`aSx$2F^58VAwaD1fx@#GT57~$1wCy2{=Z<w&7OXC2}Kl?T<mXVN)f%7XCVL8u-}oqLzP%$ZE!~m)6$dwJ3C{jI6^AOY{y5)?p%Q9&i01jH?EJ)2!q*tx90rG-ab5^_<`FwI-R&jk^#l+{3mX*c&?LP{fLE`tJ4#J3IedcsgfiV)1u->myM$ymVZ{IilJy=HJaNnw9xGXMYG)MpX$Lu7D+}?<=u#(AaY*6OpmI*+6$6onEAlbj>#+GNk;1=VD1lHwsyoWwsnb@#5asa2H1RSXwG(R+A8Eb<VSloKZ)zNQDwb3_18cbVKGA@y;;4WQmMxLyo?%|FJh^;rzt&yx(~y$&w(Q5EmTe>aAdz;HnS_tIABq_b*u?_CnzTi~{Z(Ad)pR;SNrClOeQb2It$VmN`_anAN9dYmt@31{ycuKy%M}_VnC1VJ87Ypq3P<2H7u;3*OwxOmaY4_X3o4Ko*xxQ43HZ{=v9}%J{xx9Op$UH%`YnNw=2uC#eOKiGb~4W9CF|iq;(x_UhgH0Z&e=VXHLr=`vbeR`<hoPU+4BUsh85^`eFwQTtwNG|xH?g3w%)dSDlKFa_1v$T^~awPGIM$Sljl&HC&#JgX?mt!~5pg*T39#YtQkx)HCHf`@ziiveyTolC7HY$R*lS-xcy-U>x~eFAX4O6;~G7!v{mu0mfvNi=tU2Op;8M1rr5NYD}t_b&z#9CF_jx~WLlU=rAKo*_{3A;;E-SWC^|f(yYKJu6tdlQ`XsPOQtgL!{zLXj3HgR;G0%^0l{Q{`s6CMF%JgN^GE!$oiwEvtT~Ss2)NF?iSgr;b~hw2+#Ky9j586wdA_J^EsU7L!3`wJnu2igU2Uq2fYkTr=bw+7=zJwtB&9~^R)hRb$Pwm^EM5zow)D)4C|(#k{rW`PSdHqG`A|JptQt7S&JNbzNhIhuE?f00J$9o4T)pZGg7Zh>9wmt0IHh`HQH{3hJclgzI`rio{(%K%3RdQO^^=JAEWwv>|2s0m@-BQ-*34~WdYGz=E(g~#y-bGLC#5V>^L_^Xt#QfOzrAwA431)J7Nr%O-7kGG7}kS+q<UVzBnI><5XyILW5%}e^Hed+Y@>OQSeOecr)eoHef%n)91Xh5F2#1`$M%2>qwuICqaP}F$fB*Tsq@5P((Yl8aAh$kB$poM>*HKNG9(1ISvQ0Aqy(AQbQ|s%jn6Q%j-c&qUK1UA?;Znk)<XWG;L{R$Z5RMJfkjlEfcA{XCqh^d+G{YAtky%N5!%%SY9S=Ef8r6i<3@QJ@IbyNnLg~oJPMyCzcak36Y$jNN2LXN8b?xD_FAmx>06wabz=|&spApvyPnQ4+kDH<778)dUpK*uLcbTt+D$=5Oe6mCFiyw4aPmP1E7AO4|c#9RQO~+tk~%kVt_txo+eFaf0A<E7(~zsBIQq*-zXh5{}WfX+zPcJ_tUNGnaI(j3R<o|SnWpkaG>02V43qt!LgerVjeuM$ZHX4^a0oQ9;dQW(@c58-v`voSf8)yWKXi1!m_6@uz15g$%=v!{0S{|5{>f3Krq9=pwU4LB77^{m@-9|ZUz@_KbN}3oFE-A0&F{9#{!MtVbbK4w#f?Br2G^WyjcCnk;jAKl=0B^Vat{oySyFu7^X#9vIh*X`{|4}oXx@*{k`x5Tfyg6*MMyc_u~Yu>#vrkbp~Ql(HCNyT9yOfu>(x&6cb#%8kdKPZKrc;`+YM3^({89sDE&e8{wteG&wTdJ6b6>Zt^PRNfL2)=6j2UE*ljKc%l*?8>vLHqM`fA>t$1a@m-60VMl*Jxv^^Hwf(EwP>=!AuuQNt7uzOP#W<oT2cNb>iz~oPpv!D8oIK!WbPMYvSHgUU0^x%Q7}JJlq{}-Lqo~{sM6fg>@M{&>QjM$xA+?^BZ)1A!Is#T5)0LuXKJeQI4>Kl;@Y17UE9!LZ(c_1=1VRH%*}y7k*;9NBC>1t(JK!+hb-j+6i2;n7z^f=LV^b?u9-J8ja%?(5-FO+5MRq@o95Vw^o!i}bR01>4Hl5wbv3hnIcP}q!6G>YSoe%ec*iJ*L+MiuekS*wOVG$$fM7Tfm<^vcfbqgBr#>3T-t4?17qYW=1T0HYe<#(uPYmPcGLMB8J)#VPY+VuM1WC~ToH_)jegKdi3n73p2;AjX32jtL~P&N-Zin_`DeOMrBGM8{nwlU`E61&OAlM!GajJR@>zHJ$x{p7yjxs0vSE1?z^Y>~+V{o+fORxi#Or@Um}M_;madUnAu42mQ{hvIG&7)14nLKq2klFK`1w#Qj#{+sRrAhJ^`^8d&qDnz7JQ|DW*eo*onD#$!=M6Ji#w;N1Tp9MS^SWih|Lm&dM?Z0Wb-N}t92)9R??*Z~7>HFdFY5YCz1HjuuStnLAXr4q(qmp=u)CIFG$U4IqQXnC7WYImB8DC54D9I1p%K=#m!tqvvl`6}W;XkUw9_C(94qzkT*3u?hY>_<^^JqYrfS5kIiTjv+`Ub!P_kLm=q+N8AvUwmht-bNV!80}a;9v<a96ebN=V%}{k#j@XZl@fW&0`O=b?Ii<wdp<)2h`;gLo`N?_&G8}9(mXu3-nhX;|G#GQ(M|MKHE;_!duc`TU08nGE_Q$pcVH>aIC-_(be=Be1OQpe9c~!N|{M{Ve{YcF_bkF6AY6bg)*xp{-jNoE%qo`>n5WYP$P2dN*-N_%(G6H2M=ATzjWJ~e7O0=FYpIQvvMQRZ_!Kn0q)V+QzVe>NWVVX@#Exd(u^~XNidb@cZrN8K$?k+X6%csbu(&{eaqzB^gJiVwpfN$JG0Ppz1=g*O$^#u2CK9km3exXq*BF;jqvRcRwEA+siYUh)HCn<I(*RbG4%Qiw)RhxfO(X|eJaMXA7#ixsEN4Q*)F;-YV>YqOWF=S6<4*UAB22MKj?#Y7()iz_z8x3o(u%&=q3|Ii(q*>y7p8`aPD^G*Nous=OZWjjiGrgD(dmPN;%>SuzdLsMaIOMWU=;lHiCtgr{~Ed`9Yrqd#7;}=L8#kNiMOv_@pE~k9fUlwH$S=gpOWb?oGMHiVR7Nqo>wT%Tu|9ywvg3?H#(R?|eMxg-ivI6RTbj5?cc8&)tZ5>w~hOA3s*-0sEGD|7{;l&|DcGyzN4uB5%4m2%=kn-HcV+Yt{fBhR*FEErCwRt4b7!pL!3YA5Viv#vZDWJnPZ1)e3m^(<Sa*z5n%UEFB3Den9@D?of|KYGGf!ee?V4->!as`zCtz_Sbj6yuN-tK~nBIpl2+epj5-q-+W^V`owOt8A2qAi1#QmwJ?fk_KYI;dxjlEXa5IKO9KQH000080000+T?`Mno|;(z0Hiko03-ka0A^uxbZ}vGXkT_RFLQEZFLH2pF*aXoWpZw1Y;#|BGA?6qaLs*fchgAH=y!fa56@Xv0>z0<2&@B{LwFhH1Tw%d!|s!l^~$y!i^!5=*-iq({q472`dzIg%&_;|oZZ2ax~sdZtE;N3tE)CQ#rwCvcXz~3FFw2y&))tf@^bk|oMwwMD{8UsiD;4zi)>?5omY!f(c9|nY;?Caw(iHBTbrBW#YH+=ibXar(or_c%4IDk)j}+*#c0||t4>*V&Q@u;%s*%2&iJxSXZfh}Ixn+yA)+&CYo0AOmRVUtweu`LnJ()D0FGCqY_LenlkCZd#VSk0#iJduiA_$1C)sQ!vhoDL#3HTb2<mLH&dU?As<T?;%Z~Uk%|yD$m(y9c%ts=xW!1meeS{;#**smOGkCqIW?~7Y^K!mgidkN6&e97JJ)2floxS7!r^~!5-%^D{Jlq)OO94aA*Yv)Qp@V}n@kES@d>$>TRXL6>#HOf6X#t(QCqB>Pgj-#t=TNR*%?76;I)_Rd;(P#~MRY+A7xa*qI1~2){~4k<5!GUxEugDerT|?6m`PEk%LqWm;w}Q9_TcCFK(-v+6T>{k{^0=m^L$YOdY59Eo@S+}$`Ov7LEZ1swEkQ!vlD0zdRdEcCCX|k#@Re8$55Oht!6VCI(IA)btTgBQc?~SNta?&%>bV;8&F|}_y?q&=QXr821?1HFVxtF^GeLq(P=i81R#p^vRWY>RAnZ1wl{{C%S;q7-6BgzQ#pHFNNo27Og{X=sasU%wYU)5awbPrv6_`Nk{DOtfKimt0oluFJ)foH@oHF~!(8#h`5178BPmz2A=IlT1hwNd+U-NPoJOJ8FNZIjcI{^Q%|?;eOWAl|q=J`54Ffl%Br<pl2&{mHYi`v1ea;qus|qLsAePHTJ|z4Si4>P&i>9F};U~@!woN_5hq5X+sJcQ*y2xOirQ?qH4SPV#cAEJu))_Q37B62O3SPE}IG@6*5WnWHf7>`omr#`l4fFed9&P`P`Uub`OBlTzpd=KgoaG_~Kw~kh#u*d9M<f<po@tGhfOaH}EvwNfkj8S_CrT(X_%Kgvo=_kkn8C7Q1~SE=z<4&CCO3(&fGfb9o^76Pu3=`!*d%s*b2S(B6w!dd-L4YO@&%IctSWhgyo6wF<Cqa4&}FQN9ZAG!x++gMMn#&>;4}A!spbt1wBH8=pQj7x;|h^4h6N5zQWuu5Rwq+|W2%?w39_(KL+?xO1l*D*0hSU^v0~Keuv!C?%+}fBl9(heW>EoZctC)s1v0L5xx}Scl?e|7rV}TU@DVGZ2t?HB3@K<xtXx8>762lblY`%xDM_4G)(61r9LA8++81vxms9w<$R^n$19n4XO32LWRj~}R23`S#z(W|rI+J5Uau89SEe9pNPKYOnd7c3q%SsehV_pJGf~f|L><HvR4LvyrsUVk#H^7*<3(+aCC5<k{Giw~Ip#w>U_`Lu6>95fK5>Q7AR4+W3?-J#bZfcw_GF-0MP&67AX+5Y>E)9n15<Y{_peH7`$*LTYgb~ARnq%7}T7mb@Rt0tDWv3vTh}8s)03VX+)>*NJpBsP$UdSXmU<H=-yuwv~z@^0uL_w{%_WW`K<;<>l|K?YLghTDr)ntic5rlmP#g?vg#ld3F{$6XE%%yCgIox06#hCHFF`3Lah<yRqdQwd$eUvX##L*P!1Nw3agN2dTw2+uLpaeRH=3qXMs#5t`Q+Vgr|K#PUSdBCBm^!kVU7%KhpQd}($NA+D(Ae#6*2}{E^7KPiZlZ)Qip=|z&p;^vcI&|cf^uJ^#Ysgtab7Mf(0c4QBUtTa<-95K8Ss1O70j^%d{i&T&<p<+(3STJ`gvl%$}UFP9Fb(dY&JN{X4T@-ezRQ7i_Cd(InM^jz}=c_pyz5~zo~(!#x8Eg#biYIy9HWcRs&<s#Ftw@2f*|%<_lzR;{878G~JzDDSVKrSTYv5emWYhfbwyv0wtd0C^0-BzrG#3efnGj0w7Wzqx1z~S&r5E2fCV}sE_XTVx;@rh)Y<NOSzn*IWjI>nnhKeZlqJ-?C~w&(xg>FgC_$L9)mTk2ly9S-jWS(^#x4EGMdcN=mJ^iIzde^-bD?9zFdH^2CrE~+!a0jTI!ChI=Zc)LN@@AI7EdiYz^NFP+?$Q04G;V1hBQt=f&j)d=XI>*u+A!Mp9IvjM50BEw3pMTA+uY16Nj~;-*0BSPD>&Py3`rYUL8tdScLkMN2!zS*#HT7GvF+RC{C$TY}jaaZ(U3U7G7e^jyJ6jIE(WCE`A}qdr2#ZLQ(ZRHCjz5V5fbB)4TtJN$!*0W=Fs2UGn$0*$KMdvpu}#|#d1iHAZ-a{{4oHm+76egJLwd<4~=h%W#k@V~FS*d6A{bsvy`Y_^M|Cp86!Z)8tIk}t-=RyaN^+XKQ-5}ZdRRT><Z7U|ei{5G!6LE=9`dAn?M7AAjk^p7h#%l*X;Z@}Npgj@toRUmvowOxuyzJMtMYTSj*Rb6Aj3RP23=+>Fkv?d@_R;Y_jCh;xgGvjO`)>RHPd)G+E!K92MXgCqH=8;P8{@LGzRqx3ycuQRQ*Iv<UAy~v4MZD{en$-XtFqayEuBu@;4}W&FHymthmEbU21~nMW9&}+ldl&fhq`O1o;&*PL-h&-k=#eS|ub_-1TZa~adM(ZA3GCU5V!m7`w%^dLBG#UUvl;ML)}b`RQ|QL!OBMzPocY^S3YvmOz_b>o!0BNEJpLU92;hD#vTPy?4OZPP5&cdA6^$m338{PiK73BZ)8GGzNR%3u)7B%?)T1l9ZrXZuZBxyLSSZx9L6PG6wPm&65bFfMT*oX9ly<1pNVNsW&HF+;(oGDjs(>~$`*;G|!U8yl+Xc}52e6?*p_Ir+1=xB8EVQ(rrOs|w%8M0AHgAOR+8g5)Ln3zDHXa6uJsH&VB8MhtxMeUpg5mQ3l6Hl}Rbg`qJD_cnUA+B8y@PkYxXIMpY2NuRvvk(l<S;)m;#+QSXq$$ce)x!{9&R}R0z1&Pq`zDOz{HDe13rc}NxU_KJy`VS;5AOVDNS1f8#_DO1E;NloWxyfM}q{E^hBp)X>Ft*_$`sPV>Pr3vK3YXZj764)>(}k1ZoyaZOa+)C&6nHWL;pNF6Tjl1Wigc>o1;FWA$9MT2?cQvigbv3~@IF0-av;ycVX|#1b}eMnA*}<K@u@tqoQ??7m=&jnL(uzn-oplLD<XP)}-!*iES_($9u=2#jKIiknhLH`Fa{F}D%KPrB61yeckB*bYJCSb5dkx}`vp)RW6z?(L%(b_hb`6A>AX5JMr*6v|!r;_!*(VwK@b1aw!RI2aeOKP;o$N-G(R=BrL!-A=^&53ipOUO)Zce~CZ%_u((Ee|h>m-j&V$Fm;=8q|73N;b#lj7AyxN*e5)8WYS(VF6zr!G)=_qh6K7D#|%)3m<B8*hHNnWPS@4qq)}N59lN$KUK}`&+Q2+&h4M%P`FiVCUXq1VDwAb(KB(oaG5TPYjQf4w@b>%T74Oy`bNRi<`3ifubbL7&;qvSE0ag@umia6Lc3i?PB%4>GX%xqu5$sx)5w|au(!cU&o9<9GoljFdLXiCR7oeawZxw%C;4bM`G#ZX+kZF;hlxZF9Y{xuO-t0e?JZ2AAz{T<ls?S%J-IKc*S<?QB@^BCB9OV}hgja%E>2^<`{MF5_=zIaoj)8xD={5>WH58RFT2hJPocc#>DH?j=jg}YgYsGI%7PRIC;hmTiy(B`SM)9agQjs7T747b-M`6@U0+JJ-y;yc=i#~u&ZbWd2#_<oAb1wFvR4j(Tqfga`Nf|+3VK52f1tuQ~-s5g1P+WredJF&mRjUYc`e6-Ecl83Gepj>kV44*mqh;^ae1KTU?BR=iTx9C`e44{Ugz!2z6&+B+#=J7vBr_mUHb_Tjt9+3G(xIY42O~qbgL0k#!Tetbhet=u6+vPb4U+Noy?fNn$6|YH3mO~%51Oa2i-V;nKr6%@|1=v9F7vDyM;hN~*~?32#f&_mhJPZPv<5G@yG;Jp9Cq#QVF_FTJfb+}I!q;PK<d>{0{M!3Vwuh61?-5~hJ_X=@p8|Zym`8aG-V=(NnAor#LGK(#Dr0#HY^$#p?Lvl2H~ac+>c*^HjPEZODRX7mlMRq(QzbKi)ke_KZno??f{A4m?yt1@9ssTIF6v$5GLTO2FMGJ2!V?dWaO=;Hj|0`aA%^W(6m!hT^O!`CzmDtYs%pot3ixrpa$r-G%NfT#jp8*^ykl}H)}osVsEipA@w{j>(COBv&^I*ez4X8(=1nfEP7ivrP{Ajkf37spJ6q={EvvBMUeG*U9UhCAZf%>E?8#vqCjyW6+eBXU<nMgR5L%QJK|T?3_)eh>Lr~rKtIvQl$ONawd#mKuovRf3g!%ti}-XE58eO^<7`xok>Ij9^q~ax5PpybiY+3yMCCLGm6<!-@pe{9>7_RH?kcc8_TBX%>qYyp3b1>&Zf3(&vw>yQ@-(Ux@&nDl#RohICkB;XL|r_d>i19ZXa_Y|K0=F_R;iRWWDh{Jh-|UYWJG+^Q0M>57wO3?MTU%%M;3iy1N{&~O+z)vxUpUSjLmzbB^=%^SsRm+D$u~J(ns+REoQ2qiyDM8V&g=o(}}H8Fg}Hz(P{w<WjW9Yuxht8FDF)ib&%9`WE&#^+KUw*L$m9-)f|5NZu*vU)Ed3L{6H2%wxO?<c~J}H<q<2D>)l&Ij8rqpGA<bfsxW9eza>qSK*Qx`y%=rMaSrbzHnk7RChPz<$)n>pY^XC8ZB8B$r~?V?`Yy--+I#RIVT|g393K^8ermZuXGa8eAU#ajfGY^1{Fz?xRO_`s<2ZvNY@RsdU2wPe5XZ@{wU$DFwmK7<L@uaT5%*Yrv+)Im9Fd?@aO%niHW35fBWieqdZKqj>}4W7N>KagY)!-;e{iEYXy=oti?8Ilvn&a6S~a}YAsx>$7m8osJbnFo@awauuU`y4yn6rQ*H^#1#zUK(PB+_lFi}l2gva6HT1Dj3?#0PKHK-{>=spA?7$t*H%tJGakZ~LC>fI);G*2LaShoR?CUWE3Au0eo)D)O?KFP_J6^6BG(6MW^efNo}yK4=y`Cf4bNkZt=F(8}&0NteqJP@K+q8&W13J@1l!tNDt2*YyN_}@PZCoc!Uq$<ra(_tME8bPDiFRtt8H9ZZrfUC!D;hG`XE#zgs%+q3ES_07R1ABZ$mwpURP(M_|=9}eed;+?V(}69u^a9;`%#<mqF;pABlbkmTApmDjoM{f&28Vm<kbkJ4jcT={jnEQ5yu^dlX5(?J^bl#Gc=z@T%}vsT|JAl}kQRi~fo#@3`4AUIb5|Vxj+j|L<xu4r$eI@qXr}N4Vtd&g%<JglXzMsm;3KVwwZx;n&83;)?XbX&YIjFGhHWi9pfS<i6RO5cOQ<rhs41wNOa9_^28Ra))+Axpx(Ibi6#=57A}DCq8ksW-iWH3&FqTvo%_0U^?2PU)jte%RYD_a%07CCBj`CyK!I>)oFRYJBw+CF&WYl6~4<ACVcVx>wVrhS@H4mveESwpnll-+LoAu}UgeQ)~7MIEKHM}LwVT~l^D=Y#{phZ5)twtnWy?c8{oXy?&WX?}b4kjqf`BeC9-m$p6T@?C=8UzbwSx-(VfAR<r-4Nt=B{RJe5{N1V!)}%_-TFMY=qFIJ*WG@&{pkLK9*IZ|(<_+G^ZUKX0S71z7J~O040<}+WoS*;IXo%VnOY*aiiG7lSM#S_!XkA5t0q5|Xy(~b#TT~0!gC05b`lHIvC(=tuVwGu6)2x3=#ok+BvC{MSeI;K5;wJuK-1k|(+JmT_}aeAR)y6Q(+BEPSX=p)p~`{Er*Xbkjj`Zt8M(E+w0W4@4vI;7v5-qK=7p&^1n(Sh4~+*$z#w4Hou8B*bBM42xgw*-q4KC+&6E$s2puD!&N?m7XrLSgi*#{<-ZBQ1BT&@V1P`GGt4FPZ<v^XQYhD2?;&Vc#zefWoTdj=6>aG|;#}dVE?xJROX3HSpPT2Fd6_R*7Kd^v^Q>)IM@(Iu>u<Zl<r&S!~L=tD8qmW2CjWz$V`(`M0O@msriX~Z!B$FzI_8;2wuAukb-lM2-+5tU-_xatu2wl6e8;M1<1QQ`cO`|&89dQ>G24YvNxn0KX)d6bvs_@xo$#w4Hyx6jfUL*k_X@(Z-jkray#yzT*{@IEeEw3i&Mu{_-D0>waWrE$6%czzPIqd(gXjRR$L$2tF_00sUX)+&QSWnOik9_S7vZ9{6TBZ(JE+l5qQ40ijBJL+*I}sprcM|b15swn_5BP+i(Y*=^bfH8SigbDT-)KG9npjzWc66~nu?9v~L#v?@ajlCY0lamH#|rjq;i)B8?4ylt@1fLKu3tK%c-QUDEUiyL8=9S(I_TQRl?M1V8kuUetvxm3L@&Mibb#K~i>$8YTH{t>ieU(Vu>_h(n!Up5OahWTl4KFX=(VNm+SY)C?AAuQCQcbl$6$cx=1Y`1)tnCzBgN_C*{LN`zI}-s^M+FK$b<9=wf!T~`Vs!x5Zz<gmO0cTb$e2JrSIxJ(Q~w4RUOoJ^wUGok_V=R{b9@7+Q6spX0>y!=+s(0{^**qNEFQ*2Y~A0jC8Wlf6%e*zBoGE?Cu;N9c*qNAMI=(lXon<gP%tS^yZd#ikR?fgB{zjIMu7sD9BRpbv15P&S6;OGwbx$nZ>)}oT<R|EHihH&>zS6C_6d<n&8bgL*Y#-8}6N@KDK9WTmeIuCM_ybI|{=DwzTy$QoJM`jWnbEct@NY1`L3=G=rsUW!8xnu>)yLI#aHD!ZPafH8g-8;#JhBQ45R8Rk*B`Oki|b=4{MV8*<n=qdM;p;_uNYV#x)tU+0}0_AA6rsJ4*eC5}C6Kv_9Jhl6@>t}Th_8D6K8dn}%Pl)Lh_obpdf@Tf;!FBI>UJvn<>EnZ?tPml>~JEV0CVEE8~Kvvz{D>Zs%$u8b`FmC9spJW;fIt$j$;y)^wdQSnH*4mpIc*{UE=7C!U6?C~9Kp6h~Hi0xOIo8I&2Bg@R8a;D&_<AWpv0ld<O)bJiym>G0pvn7xI6>VV_5$NYMlM}_B%7ldGD^OFiMKB=24@}do6`3i7dr)Eot(9~gj~mX^oGh*jb6)mZf0F2jQ&u&0=(HM9oYUrTTKsW33N^VZ``p?-H=DVdB5k`nnsIiI|?pt;CqSBs_5}9@|-cwD>I0@IOsUUq=$fq^UfSFi%j$`lFz9CevDU1ceWi?NKO7!QUxlrd&}2pDt2k}7GHOVj+)eY+)vM*!QN_76@}QgCE$!D;0z?-oA)UBX0e&88}31*i{lBv_=#*sCt>C5yznhzH{nNst$P3)F%!2B=;|R!ZNwj#(Iohb;jHotAklgQB<8-f%_KPW0OIHbO^N44Ojl@YwQup)4Z1X1`vW@40+CjveEXOLl6A)`veyOOI3!{|uU5ra6xq^V18QC9Lotp1e4JfR(|T8{wyocpc5Oc=B_Zo0P(mEVY{7GgrgwDJymt=|NRJj;)4GoST#5)lQ-?aL$R{rH34Ly`#3$L_Jyt_WhRugZOouS|ZTPW(e?J||?Io4CI6J}*C-h<WmTw<HqQ4Q%*RjuQyL=^EGh4}PceWe6RseM@<Q2;rFU;!uImi_4i`nSNK<*Go{h-B0#Kb{A%F*4I{=EmAHDzTX=rLmC{0P8^6o-zQe+tF;)lj_}au5Kixe^(YNU7CL=N<O1R(D$YIxAf{R2L#MSl(D)46_1;SV*hR46Bqx<Au8A*kIk=o3n1(3w!VE9Wn1}zL}*X#j6E@=-oon#UEYuqo;n{{{~iTktyyl;0H`cPqAC_xRF=p+9-^!uSl+{H(0pK0Z?)ibj-{aM*Zf^;MLEE9|j+`ESt`EXaVW!idJ0QqS+w;>{#R1uP_W(Fpcg8gi~5nzGnEyaQoha9WAopciYhU$_yH`QoVilnwjR`wHX-H{a#C3!|yw!VOulbcSyslhBWt2PyB606e;i87<cGEC3nQu#Y^~G#o`a)e`cC^jB-J@sF!O{OW3+fOJI)0n=*I{G?m~i0FlqvHu!EU`0n-Kr}=f@d#&Jm-_rYOp5Wv*z~65LfB##0-vqDpq<K~6q!*phbyg_@H@YlHrd2Vn`DzHsK9){W5|WUzOs6R5W?d5TzCI(SX@R@JGf++L_Y$(5oQc2DZ@P5>Tm(KnN{H*u;madSq~FIK1wjBl^9jA2w_&3PLbH!2POKmh6#D!(P>@KXOXJN)4!45+q5Bd2{s#`k!xB?&cafp7i6#*To1ixCY=e|mS|r^RA!AK~CSUtoH*NsNU7@~`!kg}+!(#-27m+6z7^G1L4FS3itsAOidH^-veH7>H0w%|V@_WF0H7SK1pm*dDjeuoniQ_V5zM}0!%9ZX`zh9A8Tv|MKfbQK9X8--+m-o+J4W7Mv@$5kL^a;++AAbn*2>R6RGwx+yy`G`h=P-k!F!t_kB1sz<cf2#s*7+!l;!a-UwU#IrchsOdd|7#bo}3XmY@uJ}=`)asR-th|<B#|4n)&A}>d|Y^jqcpJC7eH53Y$bP?~_%vs`0eQ3?jRjXCu`2Qeldiz*0wiw<_V-G)yiW{HZz6&vT^b4cuZt2jS^4`NQD@8)x>=xyo*Qn2ufNLUO;$fYH|sUJ?hO8K2;vj?@jGD3X`2QHgq@F_kV;^f-OYqJ57wA}*%wnz*=QellOBdD2GimbPSBRH0NSS*uY64F)^ARuLa+7mI8+%q&;HQ~F06JY|ZpXGDPm#N#Zes@~(@kpDa(^Q@Vg15MYdWUR1zl%Wam(l^w(ZJ||rEV4I&_Hf{@WXsjkmXFe|!YG??<yy^V^W|mqMULfbqTY{bxyXmppAoncH<mFN;3$v|U9jJguQ{5%r$Ub0e+H~4kD5BgBbW!;anWtBuxZ8KKl1j1mGdDe=Y|MzGdT=gp)uPAMfI?C0P27~8xk>xWUkpY<W+IQO|cZ>EfW0377#?+pD->v#vFy=#zB)lH4Xw)cIsMU3;-mwFS8vJ7^#x}&TH*x0<?;fqBBkF0iC`w`^I-0dr}VKJr|xYdkLx{0p`!2!n~i1(58^bz|DtpHOmUFCyA#S7r^DWSnqahc!@kvxl5AXLV|p+ikQX2x6UKYaHM8Ke3TV2bBMx*2jCVeGL}Au`nlwj3rTlx_;~0*S!&+B6=(A=-Hwc9q5Wvjtdk<TA0b1Xu=Bh>a=NbV1PaksFfG2fDIiMiM?|?LuQ2dMU(GStSZfDyF-J>BZ}><IR@7<{c1c0|34v+@Pj&^Muw4&)SAw_q!l17=gT%4910$h1-Yw1c_}U?#E0Ixv+ru7$M%)iM%2*@W^~^--nCzktQ>g;%9Vi#R$PjP5%a@_i?LkJRLIabGASlWV6w08x4Utl-^THd)kAW!zDsEf9L3abHyQ38PvkubSn_c>eba!CL4BZuH<j$(xN1>@Vz%&3he~Y=^845P<?toK(2v)h+b!^{1&({KdY_Nc=hx5`|WF*@o{4dDOH|9(Yho-pAS%+xSFCAfjqXROsdU8=nFP5!<G_uX$juqVZdqPKn8KdMNAxDu0%T-3<y-4*DnMf4NnyPb&xWfZJQhi68`W<qC+Lc3;%bYsCM^<dm%bjpXKj`rwhPXzoLi_BHNfsD~&C!V{+%pHgxfr8y!d-2uE3ia^xQ05h?k}vm>kGlM3(v9Pi^~e{{oiz|w7x*FEp*}@wc!ZVoIxI(GfB|lM2@HVzBzANCUeID>3&yiaVux1#|CrW$;EGW?Qi#;Zv@r(zU_P`93MDV0o%x3!+r)N$`HdPuuPHxOz9S$FSi^%#32C)h-)R555!sYx%|JZed_qqG9_<Wzq{t}-t0>(_s<YXBCBc=rKRoxLP9zwZEJ?8W)|r1H0+XsS)BSO8a`nGm-I+WC|mh?!j|!GB^sX~Y%R?0itT2zExl``F9n}GB!au34p(y0>gf7cfS({QPQ<@-PcFS1c`a8SCGO88?wzAl+f{?UYwu*61%3H%Oozc1V}wuX)ReSn)7S807$q`eja0-Xd!c*-I$bn23DY|>IXjc@Z-^h|U*(|2AQ-WPMzqII7*#W54%>!!uR6Gcu6#P?X>@TDA<R!PV@MGVpY=c9-ZSgq%%$Y3+Op{Q)A3DEF8kmh-tO0>vyCyvN(>;Uj2#iGJR4%>Z9?t-VevnK4kUMOiP)FKO}7m<+WB48>FGMX8`rt7>)davbIl0<OOx=`x`L$LxA1B{YAc$=OHIu#tGW3NYdc;G6xKRD)*?IBrijp-{6U@yyGB~n(R<^tsnvE0ZW>1~!lulN7AvOS$zDhc<{r?9tbm%&MwiMS`1{t|GPt|6LZL-c`EZddG`VX6L~%Qj5?>;Ov@;B69DUgUR2+Q805oRr{N4O3ODD=PhBGF&BaSggyQek9?6l0PY5;iT5Ctzj<8|j>SF-~OA=Mep=Y1J!3NJ&$9;nO~7?BD+4pEQepV%y!<_7Xz+!r4==oo=3*9q@QNh(kf$FYiFWyNQOQXniwDFzrVvUP?rH(`HP`f;W7$vF?5BQ-&*?&i!g)#E&~;v+tWXdb#Uz%oesSQIGtW78#s7GYkXq}3IqI9A?(C}i+|zF$a_1H1geas{XF4LK}i0Js85e8dt3`#XI)J~}-<D#*da_^<Mk&+=kWR3|;50_NcDXf=jCbNp@>l(I~Q&xrI%bQcdBb-@^Vp&$WAzNbI5XfHUIgL!qHEiC3@_yuf|O&>Pt$SDkhLReT1*3>HtdRsm807oxq17t(R5Nf|HX_bEbL=Sg2s83n*YpT&iC!{y}v)P<)OTBfd7!Que<P(7QdUEfFA0GqSY72#I847V8?};`1<Eo&IF?C(7p`pdJ((Tp#`eLzKkBXhgd;CaF@ce4muW5a=)_^@IpClY&pa&qL=o1Q1LrGeR!bJnPzeCkjhz~0&lyd&J?xAW@F;jn&iT{>q{~nXBd)+?a--$`J_r{*>h7!IIX4%nU6?wub)+@d1?47EIyM@8n4SKjGd$=||Q*U=~x!4jMgRFsxZ)nANyyu|B$P>^H(_+A<9*VLXH;A+-!ogCPNQ6n^vO$>N_rw_5awJrs%WZXi)JU~Q7f}ihrklZN<%{}kv5aO}T1FTD?x6;c0XvYM+4zwJQUfez^?(uyprf4(ZCRbm`~bct31vA&LaR0ze9x`N0dIzXV$VIL&1WMRc>bU2cr+2u%_ASOxe))=I8NM$I$sm0!>$2JsdnwlbAlQ?`EUxJV?iEIxU{;if#v=cE@GSK`Ms#mv|aFd)2YuoH{5rQ3*9l`GQLp9gwF%EA<NA2T$)+xGri}bYf2Xjbo9K-|L*#Y2&65AU{VC@Gr3&^Bqw$ZRf<@{*6`;|I16E}zaus@;UXvIT12{X5E7?Y>+%~1cu4~SRYr|$LF-{Ihi4sY{Yx#U43J32-s(^ztwlH%Ge$`jkrzF+u4WlyNS%MIYD(iIqpx~?*DBg-r0%fODNgf-CGnq0O5K3v-xZCUC|jL>(BH$Y?!=<pVD(<8`Zm_?KG@p1w)*{0_3qBj&cmGtz3W@w4praU?mqg*!$-Xv)c1o7aQ&@Ey+{A({bTFFKUx4Frcw3H>^nZNxukPs>jI7CIJk9je=>ZOruT<&7^d_QYxL$pOCPE-5WS(>X(`<ZmLQ?G6vy+mW-y1g(nZ<(HbmU-KJ0ZLwIHG>2a{qsaAKBR1Cwl3TNk5Vnx&%$Th{=#uasZg+3nu2rL#E!MQ^2$qv+Pf7X3Z+a3*i#7Fk_eBUSqy)vMVY`oni~c^M}jYB1RH=H0=wH*W{8KMa2P_3oy9TZ}Y}h2?so!n6cn$*3&NDl%*fzxUhsTbo!oH32Q`x@OHJn%-M*I=L9Qj=lp*VxnV9a2fXJe);~F-wvM-KKwE`eDTxZ=fjuc5BvS$FM}8Ve5OZechkO<r&?`a?#P#|A(r+Su`g;JBnp#0h8FhSDF;OgeBv`F|HZZ*$4!AP)NETE_j>^8eGeerKMw?$U;9+atbWejQE#R~^>Yt^M}rRanNn_m0cM@JAsv)<ed{K<*G8hah(<;bKwT8M1;Hwlh~9ml*7WBVM(2TOHf^(k`gAZXvXZ7G#I_Q*YNF*$czRWc8GAZ)<!`EDTX5yKU2}rV+8Ur5gaiKGpb%X;{T*2{rGn;WH4~9EgpviG4Vq$^=7mhnNZFTS=cE@Ww4UaZWj1y#nmQV!8@olAAdwe#j8zkC{|WnB+KD)xh)0dvj+jZs#7My#5K34}cM^XOG+9w%3XJF%mFQ?5axav@{>PyGVtM>^k6Kr4=sUX{O=X;@Nb~9*hFo&atqcl3g{5Wa8RKFy!M<#8-}dY<O6m&Veb|Fa%kgY}UD74j!lBi99*G@w3}NN6T?9#VqUzg2#jZo)8;nwx^hbE3?r55I`emQdh0dFGyg4tHk4*$ulR$T;xls&L!1dwh3`UH%W@Fx^1A>45+odII9sH>CAw+AutFHDr`YuO13E6i*4FPp`uFiyY6=nT3y1F^uJW6Wf%q7a3qw6oyqVkbZsd$v5ra7^qd@;3Vssqx8xby%|;Kuo^&p~&QIm?)!7!m>93h637>bCgEse@&-jl4##je(0fHVdXqm5f>+NKw*L&8}zJ;)EP?IMJ?5QAXTUo>E~^lPiVdgClg4A2wZTp<riENdl=;V)u5a8Zltwh9begH1ttCvG~8SF;XN%psB5<{WWJRr#+LZG|c86KKv*j`qEp`7G2mi>XUItlmS;(vt7N)9;*{MA=v2|iDfI~LKNr@*Wj=pOk_k<_Jsy>GAGaEt~b-tGDC31@lAx@N6%vpxt(oxZZN)&H?^B<OEKAqu}5guF4~YYc9BCqN|?Y}L6HwOs+6jrGXMq7c%&)eBdns{E<f$jOwi9w>mIMccX_ci_`YtaSlAsZ{Sg1A7H$Yc<BOrW=UR%k*m5m}pc{CwS-@+r>$Yc;9=GkD2L<A4G8+mmv#l0}DSuGLPo<4c=~$##MJH%7TxsqoboTUI`4jTR`W)x59D}xvDeqygR9yDO2^vDX(0OE=8z*s0j@LS?6sIP@=9pti^E)MG(#%dqD(S#lFn4wAD>|5sUUBZCs^-em{k&3s0s2C}&+ssx8YZN4+1-hoK~StT_nS6&pG-7hC1z{WdA0G&R(a75406Pny6lfu0MaYwGM9?3d%`ChwX|%l*_>EmcD>?tVam%;mluh8A-A|NjyA25L-<|9ZgezP{hng$8m8PP)kV_i=8C!*il`aXCR=Qa9(p~i>7=Pf29VpLLt4sf5Vua@+6wRDU{-ClUCdfxR9aBm&_xYmt^KD&tkXvuv9=c%T_Mm*Vz8<)X}R&_!YepV3N0&DA#D5TJ0R_T@8QlP^8#-S>#bpPrGRXYu-#at{pvf<q~|k3dUiBC7SS8P`e8!)8YYw;XQCL7FyGTt{rSNAoFW6;r_Tq0+QV4$HcG%bu86mBzV*DkG^zezA5Q5@C=LxW8kyLhbA%J0S6S;YPf0V)C*BNl84xr$DDu;c9f6nGtY-fqzUP5`#eWXGWU%5XzlX8;U-s3Pp-%5Ao~d^u!?Lt<3kCp1Z}LOVr{w#c*vD_wKR$x<MVGZ`-dGL^P8rL4<<Rr=a8xMaJ6MFB0}kZbGo2Rm{2!T+7UZai@gGi)<?Ne}r}S~C!_cU|X8{l+j$%VH+myZ1m0Fb5fKu6`H12lgZS`kJ4TseS{NM?0=3L0CG^bJsurvHc-7U1A6QK9i`J41pwYa;Ph&!r1_a~6h9iX2(Ks<Mdu>jHBA&S|jA3z{?n5jIY=MhCJy+f0%7u_Y^XotZc9UO}epz+|CuDGk4Py2*Nbx=4@7t7d<Sp#JnnKo%fG%p>%r>-l8(?HNx`Dx$2crY{%hknSwpi+GZD(WFw5fZT~aZTeTm`#pg;Bf75yzaR?s;3+z-tHaWNto!w6EdY(3xW0ZPfExH<I8<wst73poFf+gdCcR$C9!Jp^+DtEK7e~f>Kv_r-B3APf2k_hP!bR+=~h#gjyOv|8aSE7upRg*aO#R7H}F&7L>0qj#KGjxY2W}AJt7fM;B0hs^ngD3SC#404t?@Z!|2n)<FFyDefA7DbJ0aCZDj%-YC(L-KGmEK!*G|TdZCXO(w(L_$3R;<hNZ>+MUx<$?J%5e2hN2B#Z_h=JPZMS5C;0d0qW`}>BDfHhgO}qwHsT#P&c;1AOnr}!X<k*YP@?t)Oa@xvUU9T!{zU{bfJwyxQe9v1bQY+SnL-9>?Jq~3;tJ2)o!?e5kp`uW6R&hwu5N?w7I0xqtoL!$Z`~GI(6C7Uu2q%sX4utF1(fLX4<s)J!mkE9f1m24gHMtV;n`@6?uLhN`qvKQP;aGrNrAT!f=5r0(fhkr%8Jud|M$4>eJ6=fV0HLIQ*i%*{s7pOLK9HMH4YIg_V6(tG-fYc69-%2vCCI6=t%<QIZ0!A{C{&BAW(a0}n-J&^F!U*hR~3A0#&`94ddHE8<Uo0{92d06q5~ZSMxvjKlA#X6RpN_`ZqC5%R3~7NYqU{Y3TMPgq%CBFcmC1<XY@BXwv4s)Y=bCGH5Y&>qK)RS5r4|LWuICHQWAq8In@($pqY>SKM^y^SflsMT@6B@_aT2A0D&FamZYb@eBA4Tp)G>uEA?)4s^RTXMq!<873~|8IGsM#W#@iV}z=FdS_9Lh5R)UeS>zuUZ%m)mp~D_l!mqf~WTKW_};x;&NV3d^8aV6#W+vco{&S7yP<`l*=0+j(ZhA91`^tq-u-D%TU&**=5Zvo2|c`+`|qmEz(7%jKyPli>$a55r-zx_Ljd(gUSY5Zl`#utt!T_DK0K!$1;^Jm*s%rAQ!Lo>{(p;wfXt#xv_S<!X$?K@c-A!%Avf;RWeffyiZp<={S;_!*;qvIlzzh@rI7|fwR^zTAnVKS*gz8_GM@gycqYIP$exzVw$DnzBuFWN6_(iukOBje!MFV{{2CW(h&&Dqk~t+NAT`wl+KS6@w@#RU*WfeP3D)#*ZH#w3WyePj89us_g-&wJ7SuT$5|<&eVDV?m^bti?dO;^Aclv<Y0Y3#62W9%r|98XX3jBDptI)>IRitrB`B|vJS>dl<5_}ZK>kkd%!vIXB(-deO33fAP*GNN7V1QtFLFd15Ce3;V1t_esPY!q^CCwZiH$-$SfvwOpq!aXd@K_lW4hr!8S3yTfwT>@yU~K@;5QXH<X{PPAdJ&NIBl;cvnNHEV~P3zgsZSJt_W-6Y@U@!xwsmfbCL}Kb3_5%Qjo^5^vpSg>%gBPKF01n11S*BN<-WGo}O~LoO^H9c2b4ZdcdfdX0?R&*Zv>F#U*%Gl9Pe(?v<rHy|NddiCB1+xP!bar(jU6mLZMlCm`x~uYiyc;B$Vs3lMRcy4b2wXK?R3V4a`H#Pzx@gxX8Kof_LW7HYh)P>-&^vS1^5PdwEZ7v8-BD&L>E&rf^acjws$t$;prD?MveazxjZMb{J<+q+k&eoU2&d#W!tOzDC<`m;<mCI$4?EdWG>7Z9d^`JZAchZ!kSd|AkS$uqY`J(nPfwYr8-c?f@Ir39?cjQ&a--D2iwbn^K-PGW+59IK@(;1WbQ16+Mr;pon>L4y@>WZJcjaTY`){R6j9Knj`|Hv1TT2gJZ@3RoI81R>_967|zAqrPGFzWNG_(E;{lOaWsGwWlwk1?5aw!pj;EWKW7^DUh)Q%%;bo2YHR;#(-so*Q71TTQ62)Uqgr8`f!88GEBR`2d8+8iJtf6zpv%L*pa6u`~prWy{G;K-cv811wOoXbQrn7=bBu%Y<l&1kBUR2JL?v1J1FFMJ$$M@=QDNs2ZiCp>^qoy2JN$=%u8G<8$_9SbhgMxfV~+=Bz@@yieTQ@%a)*q-;E&E76eoRiM5DVIS@f?un`as2w18dl3+r|IKIyP+PVj7FN*2#=+kkG{ooiKh@%G0A=hb&hO{9W<W0C6i-Z(OCJjpjVn&?4Sn37LjT{j?pZkYt*D!2g**%xR&KnGt7`InyO*3Q|x8m8c#q5E1d$rQJCl53xX>>HM@4;$~3**kVK5h%}UkvSI7%&K*Jq&mBBZ?uH2TOAc^vV>%%8C_OhJ*jd9J}YR(btJPQ_O4e|ISog$+CMv7JNg<hRH9#539x&6wiLz3$b48Sl~T(EyUbokM?95D^l{&X{TMomvKq-b`!Q2bomg6Pri;6(ow420BVC^?!SKeD=GHo5SHB@XqQ2{<{rmwv8fp-jCYUdlD30;!H714+h(Uo00#98_Y8ga<uE^ySNfZLYOa%HNpA{yO;XOxeDRkxTq6Q0bo3(3(S*a!qPa?(%W4;NN@-~}Cs<gVV9{-{t>6e2GQZ5CtFqNto~9&V7EIZ5>$jHUMt~ANEmwQ&+|5}OF&?8qMp$avwU&k+XT(|pe#U>TA{JZp>uWduI;(5gibv2R@daajN4QPuCfwC{SA10?^PB^%zHQ-V*s~03jPgv-wrJlrB>iwQkZlht*oq<*Q&~y5<eAevrc_{W9+IhTn6!-!M{1&em`F{D!L@~t21e{D*<v8xS(Q5@Hr9xx*x)PzYMmi#3b<bkplxPodBV0)*+MqD!Nn9<b8&;HTl-`U(C$-s-0GOh-eTW157b@D-j(iZv)+oF!mATBfuJqs1u6;C9{qlf-XF5a(F=V3_Xq9kMB<8#B`-T@(?}O0E613<C~V|lKvxD+nD$ztDRA9TG=*2xT_>DEBHknOp(^ch7AVF7izzqIg8dEyp1)A<-rFJb-Cjrp`b=AU6-iM_zhB&lf8W?sh$bSs{o!^ZZXd#b?!ZLdL0_Ne$^+<qlMeawtF{l)Lx@KeEcC*@(Z=O=EuTV9D_@*Qu3v}eodJZbLO4mX8+wvnc<(|XF)-C{V=Hh)bbZdF=N-mSG_E!zSSM|FMUEv^pimWrpSj>SgtpKHIILkPDlQQMumY(20_$e6B?yZ75o&%f(D54YP*~3h682OQawgf(9Ni<I&|Aet<dni#u9(m9_=?F)Th{O}w0xh^Zp&T0``E={W97OPnXB0o%$n@e<}@NDJiS4j6TCFNe|3lxTzVKY{t96Fz?52*5$qc@5c11s3&I8db5C5#OB7!Gc~6bRJU#=?SVI1@=*pih>MVq6A79azo$VW}2u<Ti;>+s@TyW%V_~$VuY=DuS*;_^3z4;O6AUcc5NGXNe{OFAKzw(nulZ1h<v{@+ASys=}QKo|(l2B5c0}59tw-5OOP5X(q=%ax?UF3B|spZjFd7h(}E~U|{%0T!Zh8npaE<E$L3HYPBYef!%!tygna(@z)QXFnZxn9>*0lN}ehA?ACt+X!I(W%uRe~{Dl?3-M!L4de@>D!l&@u(wRr+I<y;kN@T@+NVOUJD(_Xai-d)M@{z)(Pl}ubYtl_(r_a|JhtsR!&Ry9BJ%20`ZQ_)*}$R$lsUvadYguGWN9?*p^PRb57q_O6PcWmgHY!iG6}VU-=+4ocKhyY9|l3AfB$AczpIZ^iOO|m`zqcb*u~=zv&0qV3;qCr@wh^-3a(I?*EN=sTNZD630l9S&^UQEd=>0NH#z535>o1^ixQ``7QJ^s4nRd+vM>otY7P6@w}|@7DB2_8*&psX}eP~&PLTZql-&)Euj(Hmh@<8r%?L`w+t%Mp;|ZPCZtpXKM1zfSIYmjEJ0?Ua`!kax!v6W_E+{$?F+9Id``dqUyS{#F)M!GIwF-g#yR>wUE<+B+wbsze^;#e1)Xqk5j>-}5B4!IZe*XCC!Bk1EYDUMfHV^w$G2!tzFYyPaZ|&22-;Hz&_R6q2f5I5|INDXpjrPs<1OorPM(8*QoB(!c+<%=_b6aSol*7!H+8Nh=y#~BT*)#|mwrAvWP$6@ODO7Qy<1z)f$o}n>*V|)^BlPX_TUvA>6*>btLN4W6uBch;ZSdSjRCB6RyK@V{fhggvbsHaLiNo%PR<97+`(NOnwv-tyw-8B;pIK_ztI)n*}x=O&~ZeA<$L3&sMmF9R_alLCJkvBT&zmA3#z~he8_A$SiB1{@^=PjrvU{&!P#lPq{WIf544W%HSjgj#4hCeZXtRT1yCjUc=a6j*0pr6RVHGa>J=-6K`M(7{ETcqBn@$=*??Allp>RezguANWe)5{rX3;Mx=yV<1i>w`*)SWA@n+l-7m*kD4~txMcjU$qGd6U0Xm5dfGimk!g5XGHDK%lmBYL09G}v=ICZJ2=(!q$G4Q~1A#e-4POoSpkz}5=-qThaVirzWpZsWvlX5%bltyaYtuXf5{8hWs-5fyljSKNCPtAOPEjA@|d&mR2TLPy$LmUAgy+)ahPx9hqDVi)H%4;#C`HlBdl#d493mXrIvi$@fz2zu>>jKd5>Lk4ZZ#a0+D{+icNDoSI0gIB?Y*W}n420L=W4ypQJ21GP1<cp!BOW|jjp1K}PcxI?-T}PakZy<wR(Cr;qkJPHB_QUtodNx|o#!y{gL)6(+#-Ti>NAUjm))JJ(0yIPMnC@b}-*ddlk>qXThoRxj^TnFl!82H18NdId!}rGyZt(rox?$z{Yf_9pJWGI~{m*=np3GA5)3aw{RW7Qc5Zm?^+}Z^lzL$F-Y;tkzgO|5a6mYdVGr8*lbinjZx7;1Rm9VAzz)7Fu4#tWZqlbZ)l`$?r7jG`j+Q;&ZqtV{Pff2YZ!f@7!tgr10ZoijTOv1yLFJA%(yezcc{gYD>z*TMG>NBh(T3Z8JzeMphFdd+H$wp-FL&?C!xz+o>!usryb8<^Hco(&Z2F>=cHyBicrbh8R8<TpT6tHtpB{4>G?IpWb2NJ&{q!;jSg6$o%42VSsrhpbJhtN?DnXt0eQA9L7I;_eJ3cTphDufN%(bloIjiANMZFx#Bi7{Wzoi@{jP`Pl*5dIWS8?sNfexONRPHTQWtFBR$5ZjIX$6TtJ{@5B;U&a_hmrN5zW8~&S9!_L$G@k7w$bs_~I|oT40f~#CS7GG5x-Or}E8Fw{ee#Dj)x%Nx0TDz<C;lBR-Ugwe^P8qMIr{SZH=NU0Qh=_>1a$>nXit%i)F(sHxu>Y52P_V@F?xP;YxF)^-Y_Y;%mE$!4vQCFZt&%t{nuK3mZ{|2ll4$$x5oK&SDz&3W%N5HqQjGFb)IkA=!SI8ZhP153ciO5`vmkBSjfy)TH0bNDvR>~*8sh{BDaCq17rCcnCjiDtAA~pDex$^q_j&Wy%D7~N+gQfEET}M&Zjx-CY2d_BHmQvtmyXv#Cf_H(+<Vlj3C!?%nVwSA3A0uz2uW*Xn|qNbfTd9f`YZe7&~aG4yHUsQ6aa(v}ImkYzg*tVKZo=uT}8P1^LXjio`?%u@c;<_~zCt9V883U%RO%pgIhc0i%r8b}}7hC~V*1yXHjEv@N1Nk#F2tu}Yp`ej1Kh@@EW^F%;>dOPcJ#YGyjHnzgrITjgw|h01CSA&=F+opw$9JYk2@>$;_I&@l1Hie`25le+(aD3L!ID=x@Cb%@|23fn`cC~QpA9LEbahL=DcX*qp@Q77mZhZw~=PHDOLzUV<G?jRGZ4~YXiv8W?)x})GHszv(+k5j*~XkXz(e)>w>fUcL6%SYm)I66$k!7<wKIJZ5&M}4KX@--hFZeoCngH2%8JKLbZs2s`o1v&Tuz4tVkGX&}BFdr?e9KmR+FCRU<<710hel|@3U$?0h3eyN&2d}l1JO(*3KI00B;6s?mkH)}qU>-w<_a8^}eVi?UHG<->thDR}bw$VEFiAyq4n%uweD78tO&pj<GZKw;=D&6o%r$M*QAx!my?q=R9q3SOQcMw=HcT;<h;7h{@Dvm1EW+TOJ#y)}@99dNb)KD@`m*lM_E)_#vBzWOP{r|JUw5`|G`y}gJXIw)yq3Z3P_H=I9B_y10yZQC8q#1)&Ji@-tBadMZ%I-omjNsYi;8^0UX=mTetSJ>zx9l(Z*7Amj^?%mZPD`)OZJ2pC9Sg%HPc<_w}za{!aR4w<S8-xR!(_2#9B-;-$X0rP^NH%Htz~aDJ@iD4^h5WI`cMq%66See;b;$D`$JTSm9*1^^&yB;;WwRh>j}QgK5C}4Q)=wmt{K3N8&a6keyenVl1<c%R~FTmMM#^^9;6_YBJapDnWVh{G2q~@By`rynlu_EC0N1vtrJ#_tf&kzVw~Ejx>bZMu89HVGo>6CkD8aot-K2-CJ_FXwT8!ViP{=$jQ)NX=k0!b3IkRl$lzbGKDlnW<d2+W|5`<wu0KrK$h0K<Ov7}koH+MsV3<(PjBbes7UKtypyvHlQV~j%tw9btbFNBA$P=I@aCc+4@7-6)CG!rcdOs$3`lA5*gNtX=4q{*r~2Z25D|KKxRGj8wbdZqgzE!!0mgj!MW_hUp+$)@MJc+53_9nXQ^%U4M)O8kd7yR;fM&+0Hxf=6Nc+V6=naRavRoMO%e$XTA@kho9O^`=VPaIxoXUSKCf9fa{6eL;zd?p5P12Zn9Eu?@e&5`e!7#m@8NY!J+PhXO#0NT`Ia$un6Fk_wgl!;HO|a-ty4fnXWSN3}i&C(6J6q1%ZYZg{joAQMFj2u8;el#LAs8ut+GRgpht7k=6_&Yzy%3tkRUfGum2sOfG~LqF=eb(}c-16J;d@rM)Hro3Ku52uQSMrX*pIXZRSLCG4f3KA^F_W+ml<-7kCICbfP?K43;PwXji-fVI~b;JUf{_UEFK-NhI8;nXPc*+Yf$Y0*5+y+_sQSZ=5q<XL1tGJSs~`rw9Y!t_S@qvXKaKVke^Rv7vwWP(5AC7tpLgL_V#ePGfYw&<Quk-`hiP!c;TbH&uOkN%1Q#eZ4CFwHSP$!qr#tJS7~ShndLPG@8jaZlV<ddC0MXTHufB<z|yqmKL;dY+%{4S6MF2b$1D2QROSn>1x~8m=rzwM^2=I`^T|YYU20?-dwNl^TBiDks*F5NRehvaQVkC<?SlYH8r=E{5>c)(ft%o$bN`E0a;nMX3y1f=5_4!2oUiIBY-f$?Yy<`1!CjM{%#bL%oLfK;B>jk6Z#6*Y><aM274TN##-3Wu9m%Q{ZUdr8;zo<68w#caCa`TuXG;2nulkzqHB5grm^tYoP}Y_*^x|qEBQ*+|9=?)i`})eIOC%CYzh70QkzKg5Z{|QV(B-2X!)i=(+tNCeU+|F2zE(&SU*Vm{Ja$B%T($YJJt<Ng!rMex1zb8oR|l~nJ3{?X58nM8xr0^aIwMDwG;$Gft3|Fx7@}6Is7{?0%j^4!9PE&nkHyQEhbrmA`?tTtP8XUu$@=2*NF4;@&Tp|qZ}}%6M$gMZ7O_VsMncY_&oOy+&iN;1fk;Qo;*uf<a{%+Vm&<9zVSz=2E(r{=&qvL92}MU)N&XNN4lQX27rO5#wP9vo(xD<<N??Dl`3jWMopD%>-V85pC}eP4iMUzW)%>O<5pzbBCfsp<@K~R4iO2&qT5mzn#tGxxdP|T({Qn1ns$?D+1L*}51F{bQubu*5sUi6cVq-EX`)cxwyk7d!!d|`^FJHWJVMFEH7kc$NZf98R2#5+g0^O3os0&AUB+1POBy1a?fbe6u{s%R3`sC?dRweIC<6sg0RCfbyw>DvoGSqzAaDwfd!nVv-Bhc@h5x4ZTp`TpWVq4d@0vyqyCH7JCi0Ny1I}q3<ehzxBhL79OudiiziFi2`?R~cPk^gIS8fg1@a1V-Cu>Lo&2i2GCZ2Ro%syt~P2)k7M-agvhR~SUTNZ^PeLZ@zrdUIDSa1Z&u?H<bYSFP-xEa(E1a$4l4$421T%UyMoyNEknOeCF_C)w?*H|8<`#s4<;<j!&L=Q&}yWoLY&?Q>9Jq-zj=lBHqY&Zt0(jOI?)VAoMT7@tp`7=&gq12tOWP(yCsp3>E*j!_j9{8DO-L~Z3NQ{Uu%c!}OrWpzHNBd-zOnoO$#@jy`YeQIw#)>+|JRnvuMYoF(KZ+;+vwmOHk8{|;7k%KkQc)Ev`+h)2k5rNxeEK{@DXnq-CVxOEtLVTt(Z@{OW0-L_*T-@-8KdXt1-()pP!ydb;o+SLALL+@SZ5b+6`R$vks3a}81_F0BykaXX7P!UX&L-KKPJ}I)j7A(Nil@j*0$o{ftn|s2qO`U7te(96&F7lZTEU#0pLhng6VJT%xo!F2(uC4%CM&Vu=PP<r<(oSm{9~sFjA^=Z{T}GmjdniG&YYz7b|pn06o7Ox^foc&(4sUM93w>zHFUQx=O;LD05C^tLAwnCbLzV7uT1;Ak)rljL;G#G{S~&q^4ni=`zzT#6xtAYpaT#$TqstAgV^Uz-6nTB8O#^e<mdtIftDDIWx1uk$vDeuzk&X!;$=r~R`gZ{e)NrntH_76!bAl^HHDwGijk7zwd{O3on;tI8Mz4uG2uZ5x#6p+&F>l+V@hcM<utY_;nnn;mbiT_3uhNG{AwE7?aQ__cK!GfcEg0DpW+TWyGFl-loP|s)a3M9CufHzr-zf%4NHxGAv<RA@*^k&{hLbwCFc>qzw&nOMeJkR!aiCs#>9F)_BhJe<0uBH0kiwKX?Gugv*CS=tt2o=VsQb(yCYBkl)$@em2#K4k%iYwJUQnNY0f1*;KcV#0n4^=zq94N@2dA`#*j9R(rgNJxd-kUgMW@-8M<%2%^%Q?TW+znSsX*V{%2IJ%H@Ax<GDD&cmU{emS-}quR5*dHu^#)Vi!QzXi^zx%AS!(E|SC32oK9;E^B~tm>#0F0-AdPv{@)G`_n9&6Kg7}^NjejtcmwBq3N34R814qXh*ytkGTJ27o%b|&i*5!J*oD=DCmaBMqW-j>fDk$PDhS&w3ehNX-?4yrgT-Nq^l_^-taC39OnBY*}2998DEAOy4<0~aK+cd(VcKk?L5UHinP2upJs~;&pa2|oDmeEW5sfjkpW8f4X+^c-E%xZz?~n3AO>a&DzG|{&;fFhbI+kz=cd>9tV8f+HTPX5F06>Kh%wvyju6^BVYmn%gI`>C41Qr;F1q~e`zCw4?%UX3<$qB(9g|&R47y^_nZ6Ze!5#;VSKmbAr^l%A+fO?W9^5BCY)ltw5@dBT690X?Fef`XOvhZw?uHd+@*>Qx>*WG78~nV?W=_)3wLF$6VSLEYXszRafV^&}NK(UWxl|X(hqrfe=@SK73n?qI8PF`#t5>DO4Ry+NKUq{X@tv$g&*v4Um&F37*?g%EIArZ7;z)%w#9mtgJ@|2G#W+YiYq2MW@LzNm#HS}u=n?Rlc91;}J2|p;|3m{p?a^@X3oR^IlG$Qyf&mh;*_%d9HfPt`vft(Eh4lZ$HmRm#U6(kAvN`rpQa*_1jh_!+{(SiJ2NctDCvWK*BPpznHgoSDHm?{wg3@(uj!r`{`&%66mo@uY;6%r{aqcos8CWTzn#1e_wBa~mIKT$+H&2Y^IRg9(T|}`6O)Wa!?5b(3(Dvl{Z|kIMII>R`3e=fP!z3XvEb08{lZBBVq6thkv>oeOg6K6ZfWIxWPqc`EGw$lJeH_#s!;eqXbtcfpi|IF!%UQYYIV4ABtQhlZ{tY;hNSB9APl!yvOe~y#`f6E7Xl}j*cpI|6I#fqDjF7opojfvhDQX@{9w}3VtV%jI$i~*Sp4l8~(Ty6sy9fUpJbUwY@Y~@p@1GCeS;w8aZp<qOdNmjfS9!6_%K@O6mPrJ~>indHQdap5;Dl<C?BSxpJ1O${u^KJ<44Xy-=pDMw24<QRcknKra)*^2GebLBwvNe5mw_ZFNBPr-!7plzGONjn6&lb)_oC~IdbFeEXKSD*CNL&-!p^}0FVGcwNjD>OF<MhmPLXhJqprhtN{8jp&0XXn3MYUriVpk(M+uG{jv^ayc>}6HH=|PTI=%`{>ux+~zHfA?#fzoIdfK<avmyxT*z^E_yLO{!THNZ5B5GMdg*ETjEN?yMFv5sBy&w2=+))x(e%6xTFPE!1Y4N4wAtja~46~|T>7Ig`N5_t}x~M$E%OKQ$6u>%M3>i5E>r%X1r@RiBPQy_TWwPW*ZSCn2dQota0Z>)w`dqe#Cfw6XWjdhPA`}XhLLONfFgf8We;K@^BJ((Ll4j@%rtnaQkaVTRJ%*%LdQQg8;14ES)7pPSH+4XJ#5lgs2hkWW<i8q3vL?EW!;`8;VGRb6sna|!gyQInNdX7o=ijit_Lasr7;%8!Jcm8fZJHC-@#s$bNmk7=*mWq|etkAj)*|o1{vcMnG`=59my>b6JvF@S`e!?3OPmsU>KfSC;Sqi7x6{BioYunYD*#o^Z!cdS?up67$^dZ=)LxxCPL0*+MA-4Lz3^|=#>0w*u0nf-cO1*>b{tD@$Fa1JmefwWc>)Dpjc`I>_OKQ`_UO^j0?4(QbU~||cMYE(E7EYzlQ`yNa4hoMH3L!*oLd>Hs<{pdMBA8c)p2=WrqN{z;2G-m83iMupdN$`{0~jGf*3rhb|3_fUSV|soU_njDt|g{t#lfybf6b);P7_c3Tu5GEK#-At-_seN4>e=M5uB?mI(1HvIQXpe%&d_0^HrjpJZ?B;ZJ?<iGScL^zc5zA9!)|6n~%=xyBz9j}U*re4$_6q{IhaoM?okA4>ThSo;#=a#6>IgXvp%B(MU$q)J92Fm+}~H9_cH^i|MU@_oth3M?6MXxLjFaawP#>pmUt^1B1Ng(6cO)o%2o4g*ERimg7pUe&3zPLu33nU6-D&gkC$Mi=i){ca~0Vv7r=NrJU?gux;1?Z<$L=;?b3u9e8I*#%x`!0;+on_bI_E^}R})I?mX$R<lr&&fIj!uOiL=<w+WQLiSG{K9Uf*B8Iy2k4C+BVn&G#B!r6B00DiFV~|H4P@*&Mhu~#^yS9r9xyQ(pO<oSrTIdG<5tz5fn9Dur4rG{w^r*cvqp*e(n*OJW|K;)A_R6`tZFf=044NT7uB+sb9TCPW|?|}*{Pw;3@;t&%vL7&m59&~feHsjJtI^f@KAE-{i+;cXLoHQng&^0Bg1R5==h6vyYof0#K@%pC|}lDF^N0kFw4f!2Z<vYE>f@BU6m+{6kGv=%NFN(opm@ZRlbw~3(?$SHHs=4Mvj7Q><s5;EFvZ$CLE4YfYYq!Bc?jA=jB^~2H&7}kt6dko8mQ08Lv`8#YEc@#qZ^{VA2e+OX*}Oqws)sCKqt0p~=e&$W0636^^Q>^_8F3SAPatXulq;?UF_{K*R+U1-i-cb179|R5ZoO3Pz&r<;_dNe#Rkn3Dvrh>Rm$RkX*IpG-h$R5JbLVYi`^AKzHw)%d5^^R&#^3DM7!3$+agdT%#pO8t97)ts>Z_R%YQLA%7Z14GG!sr^IRv_d3k)@s~<(r@2I^L0Ux0Ifot|%TbQ#wGAFMcPg@}SzFM5h`gCD%+)>4IyRuRV}oWF_Q`r|H^i)g7PfXRSD^9Qs`MlwBBAwedAqYwfVzqEWwc<bSz^98WpI#HiLvivf=hkm<aBSOt=kdR_~Q>%h2|ytLA@<0Z>uvS$dOpGk(y$At2M0b85y8{&`ak>GYm}IiXGg?yWo0k@Uny#qd0E9z9&2TBjtqM!V(0)nJ;+~TV^MND^etJsX&e-b&%_y<I+y0pb&V{K1!m?w;;*ttJY&+tVD-)D?AU>=fLP6;_N!8JrGy}X#=nW)7c(bY}Xc;px4%|?%k{Xu8|gbM^GGZD0)=u>^h!%A!&;>d<PLr3S(`Rb&g3+!^{WXU&DQ{rgO;_tJu*42lh)9D=?_R3tbF&7tkt^99CYpbZ&iZ=Qu>6RNE!=z#){&6;X#|-qhGYN%a(acRQgDCivr(VVC|KpcjIY4nKMUp=;Qyc9HhswRu)>fJC*o<(2nH&Q5V|-3v!w02Y}F@R}a{cm0(EhZ)eOA$Oru=C5VP-ZwN-Y}tqOiXqreA3hl(p@Lksh|h18LsFXg2TdNmTlFr@)bq$T5L9y%47E85hvmfAA?gZhELiy3_27r}9GO_0&Wc#ta$0UgM<ytz473?moeZ(IsMd;dt<zM8h?UlD$W|L^PvuNp<#}kCb4P)om`#{V2HFy+$fgKYqr@6O+DF~v8&_KEO1&#9eM`nKmreo8Xec()3?^-e>;T2Blx*k&hL0@MhPxIpH;cGT2zQEE$x!#e6yj62b#H+60@Vb>4QKd*MRE`8dus&$Y-gBL$`eQBI7J)`gLA|U5`<|SVUP~noj-lJ>#N|d!@c~Lw&bWsRUuTCQU%6t(yL_Sc+4ms1BMYpfZ;KkG>W4!1E6H4i_y{e*aq_TR`Zt~-T1Y%8vr*ctkzppyCxM)@&yTRsRL;j$WcNnV2^0W?hsuBJBLA_s)bp|nyHpR7iO><;@tIw)IF6NrpoEce;5_aD&lKqrf9>%?xx)uHM&G?lEUcs#_6n;Lek8GlaqIf*6L|K_W1$n-kgLQ;F#44YVZ#-e;+Bw+;@0O;fc@jP}8PCnoxq`&L*^oz6!{wt;Df+#cP*<zlF{CByu*tn$G~0T*GUw&AREe@Au#|!)z#Ypk^lJM~>$F2^!RW_=LL_UBp|$av!wLYyZ8oTAcAg;ZV!ip~te>zP_Am|8aV(P;K1UV|9=Y^H4|51$&fnW&E$Kv4=H~fc;uP1Nw+CqV$Z27s_RV)p<R1-CAF-;W6k52s@j!uCv85lAMu(YpG~190Q$mpbsyXnJvGYQLa%EZUrPEMTe)<;p@2@z*?@EAKL;LQnuGU%x9}o0V5^IsMoDl9)TNdI>1Y$%sCtAVA*Xhyj~C>DcE5+9AIW&!uyFp+5vRB6uV}N0_aHh6e}n`=<2}+PD`V2xGglnV{{ybO@FF^E1<7SqnO+fdJxV*B7RR~qFp&QemEK)AALIR6U}}iBj({RA5a6Vv#dmGC+74WqDRO$gMBjbMKqIME~jXr2a?4c8g?qFpd>>Ly+CETOM#Y4^rfC(4x-c2yodh7Dz<gBb-2>{%1VDlVg{~J|0(I|Z&I$1q=+)?oy%4ajLXC*T}EGQ{Na)-G1k0=hJ6^&K3>M}I(+nZAI!Q5W_=x)fB-*`n=TiY-*U6HkYH}6`8lxTVey|#5goW8*P7FGANijJf)h?-2x<mq*c*QNPI^9xUn)A1#;u_(@}$^p753K(+y92B4GUVNOCCszU~LzT?c(%$Vf<%=>g-6%Q52&;cw=cw%mzjOnRV~0r#A1~@@T~6er&ST9MILe*X>2uZqd*pw5_517g~sZskRl4i2y6W!Bw1SM4X6N24L#6Y_uH&;&+JnkOHzU|1_UqXVdF6^@djdS9!}fahLC);QyDDc4i#D#_BJxNH2RA!_Erzx8zr=u<BnwISXYb(i1aFm$Ow7oq-0_(YC7y&!0DXac5Q5XR9nDg22Gkper`DxYtU}LPAk9#flB;q`F1^(P6cAFv*KTmQK{Ug1L~|=%B1@<@g2d%k7_0+*sw5VH<VqOgu^(Sa$#|18h&f#W1`&p(fjbtBpMh4v-rBW1ahLR_5L{X6Bw}X705B;=Aon>?6z{o#d^}6juOFX1R*ZMK=|K-}&BQKqgr-4{E!mFPChIo=i%K44qKRRIpBGguWq((NWToyc~q<38NOG$0R2CXu<iV)1^nwNtisy!H{Cs1xZZ9W#ws$`A!CI%&ctnI_D%`2E<`>#0%8BRbpRxE0eO=<1HhVJ}|FUqdI3?Rch}Cb%hC<$#Qg35a;P7=GI0x1k&h7Q;bYdE8hZ{Afa<hEgDK=Z_<XdPlCGeXO62>iXrnxu|Roo?P2iZ8pX71TwJ4jFzf@xY^>=ZS?Z)|4DTvO$WwBCufheD!y<omofh#dk+$aQx($-r1u(}2xpvO7w2UtP-NUd|-E89lnl#xBFfsE*-n;4{tvO97It)sL(@mDeEw!ol*V(Kkb?!w=>Rg>$G=Mkiy&ZysP9x+r;m1oDfc(ZIyzBrE15`zEhXxx89D#`)64LB69-Jw@)L7xUHb}-DzO&)S0xCoyxzS2;v705b*u(QozNtvAv>3{s?zRrr=^}@JwPd6WN+<3|>B1ea(IZXQ$0(hcc7%iBQ{FhJsNSOYT&bgp+x1|AN7>*!A7^e&^S-^Vy{b%VQd(kPyhSHC{EDTea^BLf?ZN%Ew~`a)A2<&JG_ErU2V=@I(VWmJ9W~jdj#Ftj>bfp5;Mnl`=VLmzSDnLGSiuBdH=q+A48=gd!BFaXkuP~9bPau=0x{F8VSXZCIHRFT1HQKY#Hkk;7#9XexV6*p!ja^C=t%=eV_eb^f!2F1GcRN?M(%K^<T_#=nn+#>%ZvpNg-+qRR*=6DM%JQ&G3C(O>e!uLxAg~|74jmxagVTkBApzdNB_9CM_916N4MVoYyaoN=iTn$%`eY?d;KEf^V=8y{P5!N*Pnkm92`D<^CA(_gMWrfT*R@<F_As}NW^Uw9K0oVzK*oL-S>m%@9GHXeK&f3%%)S-B1g%lHu)#C+6+#9aO~FNXv?yv_!DY{f?UgcA6`gg4*rIBu1h;Kv;uA)=u;E+3(V9k<D05L$_T4eAx+7DG8HTkod%so8VnX2YP^KTwgfWEXfmW^2t{_1jxI6l6DBg=0D)bOH$a_AfKQC`HU8w6jm8w;Rqd!}v~A~@`*tk7GgC2H;fc0PlzfRWMFenCcoAa=3wC%+1vP}uk&9{!3Pz(z!?0DO<_+HM#f#|EsygipKHCTS0^A{)E2?rpS~-Xip7m%9;I}zbA>U?KDU}kHvMNTB(#P??AInpGj7>{dPm~6+n$0AM;QhLAZYyfS&TrKizAr5l68!ge@6p%4$TrAktk^JfOH7^D>1e?~GR^jSmy<wx=bMAqf3N*v)I^$yRvhF0>!-hxLhE!pm~$BW9sHmZ^{Sj<Ks_>5Z%dZtD{w^oy@?`v?O@2-31B4f6O|IkMg}fNno)Fe92IFkQ#!o0MKBqDl+!k+qgS^EtZv$RgX*LG4i}N0rq!H1;PYC{((;nE41bRzYjfgqb5AXzz*)#&TsBFvjatSZ(A*KzBi|-t-{^$GDaxPei2SQZRk50t>XUxLD+{tF=~MEzN|Hp6E<sCShFy?9i4lUzJg&tN>fac_<P;P9ne0YrvW73cdphe2$9YaD*o=&bi9s?Fl#k2i5^@)gTrNNalxs$3nXy{b*SILLjFmmf&6;E<L^6qx>;y=_D2{H_ia_6WJJqqwVa>T}_}@o>Szwj!_-Nmv$7h~keDB}<%6FhtBvaZ${Jplr2GN>!5hjOdA<cP%6DCs*>AbPzu|CdcH*UmX1P#s*;t0f1nJXY&+1%9azR5CAeK(H@r=X8cD9n=GRe6cs>Tx93A%|X)JNOzmg*oQ#a#xj;<ro`vUZ^zCN?ACQIZW3SMucvjP?m5QBU@zxS*>X$^#vyawz7r0{lF*}*gXCVt$tH*7!29sUVs%cdV>|V#f{g(ZO<=<P^;VR_vu#YK<)6`uP5$Bn44Wb-Bw4IGz0?-U4i|ODRSqb@<v!Mu1XVu(XH(^Zj-nIG|3*IDZD7$BCI(&h4__QjP(nrn9OD46r(9~O3@p)kWn|Cl9m?g6u0iXIpwYQZUy@qoKwU85T4POFAu{PN_i&X*Ojw)qku;@3{-r>Vy$xyeU<Ya94Xy@ZjO~+%hA&BE61y;vh_!!d-}>T_EDx9)bBiEA_;GmlXgC*kd(kyDfxyjwf|pGO9KQH000080000+UBI(kF$5d{08(E7022TJ0CI44F*aXgaCKjEWpZ|9axQRr<vjg&+cuKF>#yK4@2*rz(UKfHj;_jEH*wlr+a&9_?Y`6bxD*LVtSFKpNZCr(|NG4h022I=?4;RuKb&4-i6CY$7!2kU<fFeF6+Andhw%}Mw??|lS4n)_TV%<`aNR|b7a4P1Biy7(mK$CiC%Ko0Nz8ja)y8+RA13PZ51z#8_r}Xt>UYA`FP`{omaD6Ll`$^}<7L?pZkYOA6k#A}NdZ+9&J`i=5HsD#cPRj?_VajG5|$^Kzp7n0aqKLL*vAQY5m0Wt5=a&v&%HEMSYE;9`McM)amhXujOVpZPR4i&3}9S!|Ihbt-(1Q@+jyVDgf>8vYLav&YZiweSyqBwsTS1%DfAK~@<ZXpIot}IU>5^2U%~tw#>~r*@|WVq-$YBVclq}1l`$p8SP6GvF~cm0oh8dH^YYzqqf__AyI)=GW!i>$#Ix;@w+y3bGsx2I=yWhj7e_p2>FDJ6ba>>ag&W5Cnc466uHOE1@#gh^U0k~F&aZwvJVx2~A%m1Gapy%GM9e|i!4coT{q^$2h5O>iix)q=d;9v$mHYn3a~QaL%+B68zW?OO+#j8t23~M-JbW@f9iN^&Jv%))3r_rJr-6Sq@)x7|GsXh8I6Zmt^z8VI4d+ju2LAAD<oRP0=Kka5#rezkF!si~wMMo9OgX?_*&PQT0XR(Mm-D~7?=Ii|^6pCFI6H)+>Mc;5l{ns9{QeLkWLJV7j-L?tZ*O0me-FT;@iTXPHWcgWzI^=)w2e<rdoS@G-Fbg?{=)@Mc68di`1`wy%PYJ%?Y(;S2EI?+(RkSVoBN~W2weR1TX}WT!(S5Z=@T)k`@`kiU*FN~;fZYe`R(s=U~zqU{^ln!qy_?_hGd<L#=Y}buU@~oaB=-%?#K31`<Xo&*`s57^wb_bv&SQQd~A=O+T&;TF?2Y#kDuDd&-w}wr=AyS#4IZ`9ve@ejh^+5e=`JE7D;BnwvUa>i<iu*C<4Cy{`$?!x4$Ft7Eo*J|ADUBf8=%Z&tO#k>-Tz(j2|wpe*A_T@7KG?TQdX1#aO{U<~c0k(g3kNGuAANS;UQbksEPh#Gq-n%=rG*=+p?3e2`{IQ24p=`u(}n6Y7Gq+>MKkE3_!LqyYANdA6JM4EWUhf#k3(Nz48~8oTvbnj4ozjGFf%%aTk$rXJ^Vn7IckI}W&}m{`07%>&FaZ_iH6Uatq6$cPdzaLa9Nq0*j^y678&=as?}_>=h`0FHa!K{!s`9i+?6*==sY5@5G*VT3%4LHEW!vjE68g3!<VV2pys9kCOdYQ^kJ2r(>vGVcb}$^4I5yZ3bii_}UbY@Hs!WaAoY$Y}ruhBK~a?4XKwp&tVL*?CEdn&a*tvp!J)el4ij57b@_vBoLLf*X<5u|C|eaJkASQp&Da%#)1Px^A8(YR2F`7!t?{xm${pzBD0c#?>wrqG*}yHf4U!0;B9~h|vv!M=niL<nDrb=&x|pkmUs|v>K7Gyp&Pz$lI|D?$pV&G%&k?aUOcyoxbuSwDAFW9c(Hl{Q>P{zrsp;O$_KFkYbW;yh&oFcVt^HvTDmf?UK*ghP!c+ZLHfmdu?Mc)Aa_y(H+e3)>***%H>GdvI9LR`eyt?mRmcA19LNG^_>m#VhfZO=pvQ|;bwZ36|8T3YmC_7q(&<u5tIPIU@Lw>O)0SEM1e)UXEV!t{0AVIqt;bc1@-P9mEM)j{cF16;SIENs`tPc_36&M1H`6b^c9Ifq1ru^S3d$B1|FHWWT0;aOOUrND8evzU5m45VS`BTPy<b91I|GBZFD5Bfzx3eBsUW@xY|ZTAV=JSDWfqK7yWvVcm#JP>X{jj94W8Ywe_gFk89n>_jJ!dJV}I05YciE*aw2Bc5n&p<vIABC3FL9eGqKD4{Uo7Y`YIEIS7{Q14B)b<+$Nz*X5mtg;!!zSJyhY<E1IsI?Jl{z)u%R3sZr83h$hym$QmG`wK7>>7RoKF^u<NKowr@0Fo9d|EmY-65?j$<-lJw45Um+RlMxCWGzaTA+5wKN0N(>wl2Cgi;8!JdY%ohH9aw%8bdOO^$yRcsz-etmw-COVe8$Gj#T%0?+AJdN}X-UZj^A?5>Kgs`VKI7gZzTo4sB!Hcl-ow(XvPiuFaa3DD-i&kU|t#0Y+RYdP=-D=rt5On4qPGM26!3Q5`M!6Jg7ikyFH=t{M9X<doopj!^`vK1$^Zys=}Oi0eBF`WSu|8*5x2X$}D|j3WkgylhE9N4<|U0b)SVeeBkSl?5|x0e28V-^pRWLHx@ZHUSZpz*MM|)e}n~haz{m6w}2b{+9_U$stY^g;H(^gspn)N|VMmX}*K4w{CVJ{bCyyZhhM*Qg9An3-(dtFz~op=E-fQ^h-naguyLF7?Kt+Cw(VQV(`Qg3mgs#T!hzw^n?CmaYNwHj|^@d91FaONlRyZRNJAsP5{+!_1=Zh9cN&@a(EaOG%xF8zYo^FNbyG-L}K=nbmvOMHXv%P^N#j|3og-0TjDfhi!h4hD&KakGBROS_klXww5#juiW83z>+HS#^4c+e@p69!i{oc(!{XeFfCfLo{F4Df!?rBj0YwR{wh=*y1M!Q7hIs(X(VM_2dElZOsO3P^LB|EsTR`X2j|xHw`_Ax-EQ66m0>~+`t$@#!i)SmBK{IJ2!fH2{%^#G&egG~C1p`_`M&-7lg76NNRjG|(HOhLZ@U3-xDzHC@uu}(aON866gK&*b&Uy%&IR$52M6eO_w#-tg2vF}8dh}U`dnUU@RNJqD3N%|tdC*crc*4Pexdv<-`fR###E%Ln2t`bX?u~+<O+;XD4PI{5h%L4;sg`}?aWpF>2C#rb-GJ&oDc-B=0fPQ_4g9;Q4$&M1ze?K$|6oqMnuSMZo%>$I0yju<SLP*L(oRUf^#S4s*pOTJk1m^w2&ec5n3E+bMpo%`=%B9PuJ0Axi=>jPjp8{K=iAbvv^z)TUc%k#5f6kN!lAUhMh|s|3`~0;j{7a`1^28qjJzFc$_ta3-I~ibe9#tba=)hfiaKrtI_}{brbCsC9;8aDy&1Q)$LL|3z1oy<TVZ@I4$@+RH9oAnMXV<uXdU;(Mjka(RIB$0dT%>=cWf^5NI@kD<vap^wAKZJ<@)v`^43G#>I19Mv#)LPlDO^teZNR8{y9xpnI`3o3MOv6d{abX(`$Yoxu39+j3X)pIh1NB7qil&Su*F9!^x9HuG~Of7u1F#7}PHZ)WYchKrP6G1LfimB^<IO$wgw+t_nl0mM*R2CnsSIH_X95?JPiot}cGnZ<JUm#HnRv>jP>CS6&8QjAAuHS%m+{FvUG5%}pB?Gj*fHr!wHw^wYvrDJkwIakOh{a-wGg{u0?Q@I3{xnTc)fAbv9^B2*{3E+g7{-!Mko^G73FpdjVlec1wBS#2D=yeD@;W^4a=>UPI~kBsqfd^#9D8;p+)Ke<V=H5i5vCb}r1Xs`ut&JB{Ca1p|qS;D2v3|UoseesgEVt-Oa*4E1St8wu^SLm6OF~%HQ=y7e(n=oG)7r3I}htjee#7^{09pmB_ONyL&0qsN&Wh^Q8M=;W|3j&MCL+2P|#wcQch2Bo!G^8%1wKGdpCdNU6K`n)fojTGm1bd@wf>?gJ9Q&j03X&#JEo()|_2*y_48_0+L2Afo0YnFViA(r}=fehaSIk3Jpd;7rvk7lIJ22Qd9W1G%?2<jLM~C=zjazO)*|cBw0~R%xkdRRzbU5243W|-OR3@%E70tpVte?Wka2(uynZ9WdsPcI#3#<}nwjm3Av+%kR)#8k0uS#lzma9+ILc5iE*6dbgxMWIP6*D`a%0Qq5pU?MzcMaMGhp?8F2q4_`Fig)ZAnGyBri?LAZ_I5`5N&8YN=kwEo|brmaBCrA*&nO9XwHflb&9)zJ5v(`D9OwQswlasA*u*!pljNLgu8<Goas`Dl7@vTsuh^u{S%fLJO^m_=X0!nLjy!6VIZ%LxcCNa&&`HB=3$_;O;sbKUtlx=g@ssKITwTqITf^!k?p*TPQBv=frW(vfa<pxLkYw39E|fbG&D_*k5OAx>vpk)V5`{=)IhjpZYIvHz|a#c5*q&`8k^ci4(|o1@akR$1f%&-OT(?gAQ1cj^_PBjM(&X+v}r{=d6Wf74bhwQj5+0Iz}Ez5*z1mNF;)&p*aj114c~f2X_C$ueZQgksAr~EO5@qO0qSGo_;Jf5eA_TJU1`yez|^JN<;Xq)qYFzU%zCGF1;^H#hf$dC458CQPKwVK#vG3cXxKQ*wjc`wieL;-bXjs6ZV13Y)rgw|?Rc$V$}7f-I(XB@R2)Gjh={B@8KJgnbxz~kuFXryb-<5$Y0`jb+n8m9{q!><pqHk`VB`$hv#P3~;cTFsv~fmDkuGm6hDwKz?1qlhs~6}s-wP>Kng+KyBv;DPb}_ZTS7fA2DNVoR#fIIcR<x-SN8jR9;ZITqY6SgPjz{%;@UmA)P;F(9l4$3GN%SI+Vs44JtC_r5g&Z1o;BUf@8K^FZj^|0RYt~2)Y<h5Kb2pZ!mWJZg5~2{N83X}4SM&nqJQ|<%4N&SsO+g|RxW5H4vw`Rgq5)f5qDPlW_dw(!0|e<jk|Cz68Ob8fxttYHNMxeueY&fg8dUdj2U>w(2KNz4T+s}n1wv9PookDkmQH2G3Jk(?O8m)X*&#ceA+ZC$wOE;PCK8BEB<hxiVkl7zbrcJ-4Y9xLz){}27m47AE#`}ikrNMdw&Bex1S|<o{4Z0SOoySQ6;T}2uZ{=~)wtI9+!%^Q*Bc*W&)z<)gK5RNZ#51)7GyW~niP>kn`n+!+ZEXpumtfk^EUs&t@KD*j1bR@g`fs^K=+c;P~aJWrGO?toPeK>r$hsmvuqQ_FdmrPbd`C)E^x|HP4z!Az9-Lv%mgE=q^2W7#dtuo`VX)mFfRO~N(mdFhakXUd<`GvkGO$8V5Au`&@o38&SF6XxSbSWs{+S(l@u9b_3?~swxzRfdaWf+iscGO5lfeCnSmj?7gc0J7!8i$%_;=DJO{UEjU&j}`a3^j+H^HN1dLIqFe}lI6S<mVEnN;i3P}GGrIKo1MQnU$jPXXqVoNS>AKGF(sqzr5Lp85kU9(vPt0(y8UK}&@-<S}ZmIn-ws7-1#41fk*;WXkg$pM7yw>POGCY*$Tgb2Duk4sS7-nfE7@@-4d60pWFIhXo?RQRlgH}D_U)e^n??I5Qtt;k$qv%W%H5w}E15(|q}Nm8gga}dT0vz^G(cRlTK?rreYsPJ{s?fr8bo09otVW@k(i=N6av#TN#*|gV4iQ@x17<v|x1~G6)hF0J>q^|bjSZ0g`XeQ=K5@}mSw8Q2QaX0OdrX>M0kYee($XLUMVCku<TxtE}2E}t`_C5G?+t^Md%Xy*P$th^&0&I$gtKXS!hTVaQytz%k!YWe%mur=KUQfrv8b~`4lG@5`_!y>^!U|YW0C4dH`m&)JmnNyoU3?O!H_b^iOcg5GnyJ_lB#^Je)a5DjLoX6pUoJBw5C9_qfJLx3#g;HI>3kc3-_>@l;9jRbBLJcvoLMoG8vEArV4Svs*lLnTL>jg1zQH_ghciO$`y3ErW(N?p@AIB8?TYy`TtKyh^!W-(o;Qo92sR_w@94LK@`71c>W_N4R*4nUe=e&;;z|N`p3zw6tGa+;SClUT?K@j9f>?ew7+vqB3Q)Aj7(%<?(sxap1Q@wp)4!;A<+uUkKBH`oIQm(Gd}O>Sz^)d&7N~*|&sZc5tAo9!Lk}?bFaTwPw>HA~H!yh%JikahJff*O#w8u9AOwpr<MPae;b#fwbesa`<J9o@1;$)BBL{xsai$>_oTve>`moTT$6Yk~Vy+8N|A|E%$iDO;n2&e$d18?oy?H?XT#VYfJsVMkN<SyCI4yxHI}WiQ?ZEZ|!XnJD({+9Q@&r$&Ooi0YG@9{^m`lhPSXW{e0g%KC(C=DD03NV$Yh9??R)EQglMQGRk`7d!h@K8~p{ox%OQ0yNX8klz7;w!x?k#jvkNd0LnvKkxt^<s8C=+{2fq7Z{u+&jRYH6PgyH%D`){rdiq!i^^I?~?aTo9H!s@!vZ32x<?07Q<r9mTB1IV>@P(*1O$g~W<9Rc_MCl%OOg*Q!<tEo~=fB}w3&M;JuX6O)GIr-+`Dhc3^7_@%X6Gd0xpeXCYmhWnSWbuh17md0s3u|WOq72>n-x?Ck<kbbA_I__DI`cjk}U$VF&q}ye*4YmpMDLpZOY^~p|h?YmykkqV}48UfS6oRQ`;vwp7r@EmLeFy>5Z=jaBcXCjmqXU;MT~zgK6nb<FGtGYGSrPjb!in&u>evT;TRjBVO-pq#8&rL3XQgi4nB5rV^Kz*C<&$yExKH*4m*IF!?1Ha;-8rdH$>34Kv`~q@nvbJLjZ2UM=CXvh9qQ^{<y2Q*xD=K^Q>1m}16pM33P+HNuV%&P^6UUOybG6-9W2xIT&WI}R49$4tw=p+g?Ng0BNZis+U~0bp2(Ya%&Ap3@PQzTa62MRjP7f#7zLH#O`|%oLeti8rG>V!rK2vuWPpb-wk~4dR!nckM3QQSuT=M{N=46Z;Ta(%IM{%Dq*~*L%&r_?838q5vrefNy*n(>rm;UNsBhVjQFc*co{8OA$yZ<GUgZ@k(Jbd!nW*=Y5m$GtqN)CeQdO65`K!4os4)+h<gkZDQ?$`EU>2n5-X=!OZu5a~Jn^_>7S0Q*ryyp0R5*~HD|)H2&W7hWLVTjUamH5?hDnhTxZz4KDVcbf?3V_g>`*1iV&iJMJ@Qgc{E=bwGkfng)x4Pm_W(`s>qrsRWR;SJiB|(?i7Sszuzae`RfM=4qEQsOa!`y><<wlfDSzA4-)(mXA60d!qc7DLfRw*QNQ@i5HoB(%v>wW8L8tO}V-QMy!hQ4nhu1&<{LA2n??=PI=-n%kzr_-?$XQ_*d>|Mf1IAYQJT5fLCwLopW!aP}dnyZn$ESn=soh12QB>O9sI~D-kjWWoPC5A|gm#ngCPn5m7X;5=rwN!Mm#@6>$tgf~K824UqD1NP7#3xfXb{b@1a5g-BeN>RNO$X#i9&{}0AKi=e<@<nR$UMuK+<q1{J!h;dKYhgc>U&LQl8LjRL0+5onKyEylkAf1075e9K966%ELTjQ@Ox0FOMsX*dVlI8I~DLd<?|rZjnV(an2%R&`@5hhiS!A)E4=8^;3h(2m<WL5Cr$6CJ|Stis}kZwogkO7QX1gHtMf#5SvsoQ6ld3{tDA`lGtT*p1>rPUVxnRL~uY?g1~A_MYaN`1w2|YFUnUEQe0Sjy;tOUiAG=Ca#Cip#{G~Lz&#VU$F90uB^(7nyh;Rz$A;M3P)=0aV{&^@h=JGgbs7VDw?!pl9sAEPBRomLql6>~yh!pDo7$psKxGF|@b$asbXEhwzCpqlCcO~D9i<Bwx8)~R?SyZz;T!ZK?_BjB4mt@{VNq(Pp+S)W2bE9iWzYre0BKTf^OZM*p=fIfp-@;`n=~~D+p9hr7{8_}VF&a`ITVMnHr}4z;$D+TkbdapdJ(v9__$C5xOE#xQwM6Ouexa9t&Vy^FH4*&#FL;d9R)Ed`3j@t5Y(Zx<NY!;5Wba^AMmc%@Fh=@)U>89bY7gqp$|421qUgcnO<gI*Se%-xef&j@8g$zwyD~A@>LKZC-EX&7MXqzb3`CQjlJq^y)g3T5mQZdUO;b7n%gvKjW=k^d_hDeUP?h^V-Fw-b%AUZ{(Dy40NoEBo|`67=pz9p)itA?I}%dpf-5BbNZNJg-JQ;+A{A)Sveb@3NwXlj-YKP*g31A{M>-#r9=!*N;(Kbbq;EJ;FA<Wi9mW%&`n(%2%$5I0Z)A{au%#C<NU8<*#0(MI!?f|_$PGDmf0<H4>xB^d>xrySLaWO67?=#-v4ETsc1*b>M&yA}1T%AfIlxM<c}Izqyx=Y}Z~9~?T}aQT(_83uGCIlT$QFxAPMAg3{6m>9P(Xi%QY2D{n=vnVo51liWmP_i3QnDe)d4xdNqo_igMXf=6cTHnrNhHVgidyluDc)MX&(VqzIH{7&5I~67hAq43mk0u`W1^g%5QDPnWd+r*}%t-1w#8PS&>ri4;xdofN*!%>7}A)$_nP<1Zm?8ouTm8<ZERv=nKOOk7L^xt@*<Iga+IK_`c)1NSf>3fr5aByXMlLDA}ZtEYjAadr9jDu!Rt34Azuu4vyUGoZjtK+w7w0|Hp3sqRmvMJ;NMDJ<0SS_w)3}K^=mhUC5$9yizDAHYo=NkXvg`<lHMJ?9(H7V9WgP*sS3dyuC+|G~KXQKeTkjJyetbZI%x2!7W<bp`Br5`_tII^hl=EE?=Tty1937ez<$wIfE@gj=GnFhxSQG{Y0Ae($3fIKU@n6%a_%ak#D!3y4U0EbU5$2Cqq$NJCD?nY{Ds)Rh&b!2j4`Wy9jP2dNVX9AtgItTJWc$s+qbvH%KS7UO!mwI|KN}F=P22gbG6BoPEQU?59Z*ISqvW5Ht7(YKBEr@Ku@mqUYenIWLhY1F;jQwN^~!F7a(W43t$J>u1(0h>ujWVD}d+D*RnV3q(C@)X+b>4$&Io-dY5HzG6Ls(0#3%1b)7_E`gsi>sBvF2rlg_O5lgont0amKy?B?dWC`=EKkT$?<rBpk!w&7s#OrP7L``tpRmPGJEe~wS+rmes94~qS+!taT(XeF>tze}=V})FC;7nI1%4{|U|*wnfuFBh!04dr3#u1H*8LR>{M>ITzsy#&1_E=jO+XMlta12W7N|#2fg%1Hcf<`eYLyOA-vGnM>yPGW(zUP5@xU#!A~Cq9$bp}S7dqr-Z=9~sD;{4(?#g;bS;9C}!DtmQ>Ya}|s~6=*FJm02VmzdXv8RSn4JEUT_t!GIOBsh<(1V_uo>APNQ2cC{?obZ#Knn4$HiN_7dd|(N?jXtzpX(C`5Reih#olfg(i#KY>*IfE?9de*+&9+Oj2-SxA>6~rX2^F>8lg3QD}g{n-Je7d-I|F6Y?U#3%SX4W*cjY77KENlhoVe8{2hu%pO0<_aX_zbwXwJou#8VXnJ-X54qha2o+Y?f<79w`xfx8Sk-HE|SXO7m#b;D!N+n)BgV2?u(sKcmE?!~wRK1g3$Os)WL{;N%N4<8FLodPwpHCFTIHeRqe4~a~KYT#JbX`1S>bf!=qTU`4+9{D?yZr5d&a;3iz(U~4)EteUIw0202$+g6)5Guge)slLyqWv#bUZZi@FPC97lFV<QxngX0(<`#P)h>@6aWAK2mk;8MqLz_Dgh%J0075U000yK0044ub}=?zW?^%5aA9<4Uv@DraCyx<{cqdGlE3?}*kgbz*-*6P-Ya@?RGghSO^}bKiJSKTK_Dn<W${Fj3Q5U!i`?IS^RY|rl9b})^&O5tVv@T%GdnXoU%Qm!IPc$m9Q`PMxw?KKp1%8CEM%N7q*z8}CALwu67DjZr*aVIw|TJ^XT!4}2czL&_=6XS579=74RlX^QQ|!fPO=+O%1xA2Nn8deCoi+Iiqcdr#KrULtM{VV?ngg{n_XHZI5ymm#Qa{pjkmk7l*L{aAyN=*?*sAoRaA+16=gTF9G?jBy|{Qk7F$^i0Pc+xuRe-6idRx3rHI6~kbo?YqY4ntcgv+L&c#iU@3wL=__UM7y@;wx0&#hUcvmu745(eqi#%GyfEH!|c!Y2(fSRWZW<-HZ$PjQwz_Iamm8VjnkFo{y*~+L2ldO_ODx<ypZXsTseiX~R5OR;mt9&gpUp>$SynrV4;o@2>lRE%>O85JyNRabmnBaggECO-eBB+QmqGP#BQ<!?b*;bG9UA5g+@ERs{vl74!0RkpeB@#kYWBg|WA_p4fR=7BU>NbfbksHC=B3a16h?+s%Z}VFb=bKGZ(e!a-Dbi#QoX?<p9PP>|HJBmcj*t=fVh6qREWIa!HD+fUC50pAAhES@<y{n`+9ie9MCIBO+jNIQeKOJgEhO>|WKn`3148q(rfr*-2?`j-7E+G^BIh}<4S83B!sbK>Ljo}_N<rv>z+2QfY%D;ZH^5AZyj|>K5CO^R?Zx$&Ixx#o^Ar-CcZSS`B>?QkJ~$JgsRUh2?g(#Em82C!93xl&Yk={f)qoV#w5(`K#6}WTA(u%eH=wvMfwF9|lYuKXMmJFcYYrPWO1GlTHU+I7i@l7iyhuvPx>^yNPGQlM67Zic33(RZ10`7mD^!-}BHP8O1UZy>QNbuwi<mWla4SWqfN3o4U9m@{*d$rFJ+1O>2$BUwLVExlT1#M9DJ$<>lt2=g7M83sJP19zDTUJblMk{IWTL|77ay+QU0lDI3=#7pDRFf$cd8W{9S|C5c~K2uEz{DyEut-Q6z)OC(X6mUfB`oNOpjFo)}P}zCrc2e2*c&B+7&VkMY7rAe4{MOtB928<V3yOz%YHEm--IOnXK>cx3cLE9FJGlLy%>`a+k#j7qzAo&reR?UwwS};pMMy#Y8w{4hLuhk;cJ(G(gKc8V-MOPEJnZ6jk8W$1yP+sDH}Zr%g>XFY}btwq6L+p)WE|<V#Q*$<`nc>(l*MOkch5f!$kQym&SXkQE3t49hgHN?^o9xX#+~g|o;1GRHfi4hy-2PDz$jVd$1JUHW3}i#=jtvPc);eE_fN?_T}Q0Ho^LfC3c)zAHePlnVn25C=mTR#2{@t(=Z#b!)Pu{$M1qV>1e`LV`52`c)=xLk6~q?pz#2Glvy&=GC3n(9A-Tu(t1A3-55|Ybd@=(b)_%dT);n(_}5(wE=rN_QjaFFstWez62P3eXN%U*sItI`hg5h*zLMmsW=II%F94z`=kJic_S;=u@@GKcjS0t0{S;}8b)U2)X{xHt)vc;=BQP;aO%zm$3KvJp3r!@hxo!_;;%ATfVz#Spf=HWJrZwptbhTETSpGy+Q8W_?)6kbX9)N+U;G9tjcWbknZohRLBYYzZ{drpi)SD7<ExMMGd_gRUcOOu{sO~)dm&Cw0UEt1Hl>0HcVU(n8`bl}Z=_Y#Dkwh{mHXlu)AID<U3m46>-QJX3Q>WJBi~&7BYb=HcW35_$KogImdcwbz7Mx(<-uQ1n0JACHoK57)$RDeOX!gp{L_~a7{z%Z>!nGr7iwv)LUJ-b8)*3BqZ=-w5Z#;)CO4_1v2?+lT#{1)!*hu%M-=&O39&}kLd8K4KfWA{JcSTVIv9GwGaZiO8Ak(Xz}e4%8%4dZaS`sxb+MD<rq_3kt1V$=f}H?2F$Jmtmj?k?QC8$-o4KP-VbJl1EOLnWw)bM6$lErW=TTZp;bNpl@k$vNQ5CPeu@O~3SirGBmG97-of}@i2_$Qii`ON{R<XFYICbMEZ4ExTs*e{SZSEy&`!A^QUH~p{1FUsNzBuy**!>?ob2Uv>*Gn?^jcLkid(;oT1sdR<8E{SDEy(*C^8VqF8+Pl#w@J3hZ|nDI4h=)JYPaGE+(?uMnspc+i-EXp6W4EBWK^UO<y1;!pq-!&_kuHpO>~cOoat&v`nHpyk%4kxF|6rJucw`?d_1Ohw4LkhT(#_5E&HrzIP#EsSZ%NHZ38~d6_j)Fv2NL0E%k8Z*SG_Wofqh)>$c#6fm~;vFJOHF8)ES@N`RQYw(0%Y2WD@8+4qAnru`Z~yz2Gzm*|~Fv|?SBJA^y0(f8lN^hTaNqkV$b`YlF0-*`PzL8^}LK|jW3%n2A0aDYd+3I;CxRI?;1FpqW7{&W=uyRO?^)4aDYPmO!R{7PI)uYIw=$aBK4=XsuX2B92Y!}7QsqpCHg+ik(8b@el>Fncx{^}kP;+9@&@B9j^Rb!o1M0pHGO5jL?3`jm!daFyW5-A`Zv0-zx)VXSlqu>1#SUI57rtU&n(*aOeQg!nft4#RrAwlo8&LW7%Egxw~)M$R_@=VT!|SNDHlq%W3f4%+Grl5FXC4b@SbU<IJemlYZta96(g*rK<iLkCPQ$4VlqjUSuq5nAj3u%P~<T%W0sn{|*aDA1KgoB1LFNw(DBvPTVo$4TlMp*uP)h6Ivtn6>FB)iG%GsOcpL8>q)WBUCK(!fierzaA-PjA;i(3zS}yXVr;x2^nIxXmfAV%!#HGLIQLf6*^wIGb0rgd$`QHVaf_fhJen!(<#cxRslN%Is&pDbqAyF0@f;w^9XG2MqZo#JKY#Amzir3aprk{VRU!#Ug-Gplu9y<F7Rv3F@rC6AX9e@I=LTusBu=9TK8|(IQq3=*Kn`rt(*$iLkSbF!($2pAopo?R0-7F$e`w&36TpO;)v#XKkP6U=zsMQ0Kg33Uui4=<UmVDjEQ4XnPXvxa*HgMt`hmHk6vGjX@=)*sBNOYpr-8ZeQUM*w4glIdeWp<L6b6=$wHGJpw0VhvjncpP>u2wD`8?KQ!rpyjiO~LDYt79`^_9IEk^xns>z}>SXk8p$68)qxX^4{oHWmFIy|HWf(51`|8MtsKu9oSa4WTV=#8frppyH4F+{NOvcP&*h-sv<V@RuFIn@=lWN5l2kqr++JGZ&mZlR|R4vv&x;$aY^GQ-(<gVDK~MhP|#>p!bxv5*;CJ#d4RBUxzyT9ZCTPUGQt)-VT=mzA&6k4*fLur>x<{?)T-sW3usDP?7;<giy8du5XEDi(@m$skXDs0_$4F+;i%u(|)jV4DPYh&9jK+~E2yTX6YL)Oi}Gsm!1_hd>UaI4X`|++p%OBZJ2x`?4T1bb$W{vdGxcHtyROl!|_%G?3$5&q7*ihHnk36uc?e(Y6U!RYR~#hg-mLAS*&-fieBriBSj&tc6FJi}7rBG_(UvUh`yjY9@rWn)=~k`d3C(%QkEMgXp6IifkSH_3f1^N2ozm!$);PvpY8O2HJcvE3z?FWL*p#V4Av|&OW25b$~`2F*Fq!Sd)7|f)1l1j}9q*Al@sx<1*{S6b3A2m_$(^Z<Jg~0Jx^G*J9=c;_A~b*+(gEK`2`5LVA>@dd(^>`5exjVeHdw?O$oSv7tgIvE_6)3!3~eDoa^F$_@!2I7*io*%S<-43r+H48q5><g8h}ao_S)Xs5BceDAi$y6V+VX;O`k(pcnUQ_ewzcX79rb7v>XcQA#!Ew7d?$h+9858ZXaeWS54!`{jYdhw!iQ-#H*hnAVF;cFhgZVx|N?ZIo$1NUD&&?>}aM~X01SLd?(Wjt%uVW@H6ZbTim>m0XCW~(ek(x60Uf;(WOo2V>(a*eG~#|l~u(aNXsGC_9p^rQ8h4C^^~e~D_3^_cVHl^Yzew+3_&+NBPH(XM)AwXaa`lt0TnuOHs}2s|u}d=<b0btQe7H5IJ>KCGxxYgZWODR~^AAkF)rPeG$Vt(xAbYE~V^Y<j|NpLi6PRO;)I=~;Slft<bqmvEJu_Faz}k24m$|C`>6xdgK#>qPx95e=g8x77yPL|{!oZ{}l#0Cy@a^xZNXUfpVG)>|pfx~g%gikHE%Ias#Kx{AUo)+(Jff>vVwno8V+>v3Fs`*J<bCh4Cn%KbGpHoU&2&h~(^nTeuPk*Vx@Ox=guta41n#Tea<bP#%idSk&0xZUHlbt5>DdLYE(dOO1k%@Lp-o!6*1`gWoFv>y@57MOz_hIUkKgtCaWdHM^h0i`-xLJ+GYyRQWasqHdX2AjD41LzR;Xfh8x-w@U(O_a=iNQZH&?%szbzPL9+7K%nQwR;Z|#|>5N!kgI)=bmHM%mIZs`^m$|c4)8@LE{!1W0(%OGxJq}C#FvA^96ZDLwYMeOtm2}j2v0b8seJ6Hfp~2NMJA|0`YG>UZ2J30iv4>5JLrGXo6@?@pNI-ZC?Vy^^Sa(RgHbvWeE&+t$GderKe!yEB1s8Vq#YeAdN6A1w>HtE(GnDaYb@9Gj3o%vh?33$fnlz4ntwAw=2l!Tzd60nghB!#j)!Cd%W+EKz{~cDmB|58|?dDF(9aNf*jB^4Rx|6fP2NEqMvbglXBJ-dc92~h>|1riTLxgQJcpWm2p&wc@l=4M>CE@iaktj(KblW7Kk=HblfU|_$J^ZSMC(o5ki3ouUseFu-wWxf$Wb@c1>FK1%(C^HiHxoLWr{=UR^h3bejSmI@HqPOv@^eUC~TP^PBBqvrs#pmN~&uK@8dm>3w^2cQDuBx%9%vIKg>kH$}8LWSD9gH-(f7Tx=K|8f%T;JGw$cYplH<E4cwT7U7|@)=I>%pebk@{geS5$-&?JFTgV`jA>%>IZj#43#F;1K(<f4JvDK9rYCNQ%D3pl=04pei>8*FZ`UrNfh=QkHnfI7uj!u-DVH<0l^x{d;4pa46T_U?Nfz)-O6q;)iU)MnS+1r5aHb$0*VDw*FFebZ^Gtg!IiaAj;7J*dHPy8`#A9$9-#V1GlcTA-)8WS+^Goj!nOvGm&mek#$Y0R0><P<MYr8pLrsU^#_=v-%H1-MI2Pj()EFSg8B|0>r@bm!+ZN2l88kp@iGId*(g>|*El>xh=F3?cO>!Wlh%juw%g_%?Vhz=P;c+XsV0+CfK!Kcb0layzezqwfdr>#4;W4^GvCcVoYywHF{SXE#pO-}`nz|vGYJ`3T2$}D6V^Vy}A$E#5<?kL{?>}?Q)swcPHZIqu1;I`at+#yEa^a7tekUAL45UULsUUeZX3R-js=x1hMZx@O$Cww~w$#k^I>O920WY{|GBt@9$6g$prAy#=6rCLrzeml@o_1SWXXG<~IfB~fTJoHH?6MDevwYpH2frkjQd;@Nd<cZb}!DC$`0JK1t0$~Q!$}4GC!%zR3t+mIrwI=s`;JaF}!<<K@4Cy7>Z$%Y98%JsBS(#L{BECLT4!rS&#E%oK^R;xCl20|#7a41LPoG~nGuL)>Xili_L^x65*kZU(WV#S^jM^aci)8eO(!_WC$ZGiJ_t)1i-~9UQ>d@iUe-UK{NK%F7bC!nz_lc2e6Goj3@A)t$D#d0)eR_n0alj_fFI(}`Hxgv*fHP4g#6z;`XaH@Fs|}rKA0;Us1E#H%vh50rD1o5jBO3gxqJWG+eR4!_c3C}|JbSgmZz)zXUT<?)tz+T9N?=xLc<P%AEWQ?2a)hrPvJ{Tfbk}<nFj-tP6cV)JV2JfG4rU0j{2~27DAv(48f?Ov4)j^xckh3F^X}T2jhpUivPpCW*gEp-i?hG|=vfOXiz<f$FRQ0vyGU4a(TP65Wrvx(by%#{I~BV_6J*^qn;(o~`Fyv4&N`0tiA*BdZ?Jh*Fq3S^3#r5#`!kGo5{}UWlcYl;15R?2L1Y=qyzIuxO<NN$H~T<q68MroiwH;CB^|W*ljOLORYWkp!*|~y=FjvY8Bq3904mX1w0}CZA|Rx6=yMUY9IL#<7p&|E3}0l?hB5~S>x<9|6L+c=9S?>l6xQ{N_v-L?+Xp(p#hnaqqXgUpjty`$&I2Cs4?H!i5Z*NppO4WC*f-%^f)*m#i>Qj={hwqR;Kp@IR3@ce0Zp&GO{x_&gs?bxQvuggm*hQ~^hPgng#HK(kwMIoXZQ(EsIcK5<PLek;-5xeK&G8+2=FGTzlC`$oX2Pb^b14?Vn$(E!soX=V9R`0VD?$tl(1Yyun3h1Km7Xp`=?jo(-&7yU%mVF<=gA<!;1@N_PLP)O#>r9AP!58OD<{1!u8I@jm$7ZqQyC#on!T8pX57KaK|~<d&2lQJl7|5LV65_1_u1l4&VL>K8zLZ4F7({;&PhMxuPRoa67aN7dd@V0eEP)h%G><55S((*w96A8|js+zDR^vr)9W7-1s47T2IQiAzPhmMg$R^iE$GRx>f1rqQl}~HKGRe;9|rUH(3?fDU~G@y&l)~RFkQXbq|&C9;u7a6dqm~F%&@IZD$^~bb4zwc(sMtqkdwby?0+c+_f73%(5D|m5mvAj5#*kAn$7SBlHnkyc`!B{QvoLqeJ!S3NBeU2T1gR8CbE($vhT6bpps9)6gnbt&(M3DUWt{Q-|J!ax-`Aj1g3gl&PDE+^!M`Dq3@TqRFBUH6e)M@%oAI#dzK?g$x98v#K4<(CqmX0Q%zvo?@)wcX}Q?*`Ti-grnA31tLY!IA~LE9cD0iilo<&B07*(ADX~HS$7VN6vHgU+GnM9E=RpxF*b{14EcCaKR^N+(_T64eaC!`(ZtfT%S$Iwe4);oD50CjaM}lOaLV;`X#u;u?WM$L9^z|_<=Dkw4{y(pJP&ITUE*p3{&>>E8MexzugM^>pI2rGNR!2Hzn#N;t#NfV0I&WXPkXR0ZkjJC*|Df|Ks5QbvzcOm&G+7{p8;5IR&)Vj{rKzCS~ae-8E&!Rf~M~&8xrc*eWEx#!v=%WAp?3(1%}s4?Tq_jn!dW<^pXwhu?_kobhGNAFI+~e6)q#DxgU4qk(<>()$rBIZ=0NUExI*R{<d0YY7kvh8iXdIYg%-euZ>|7k{?zqZGN%XbuQ-MX4Z#Dd6;eeTEHA5ABbnp4RIXO-?~RXE#_LNySA4@&sRSbli3;8Pwiw0GQW^Jf?NI%F739Dt*BoQKs~5jK3c%u!GqbUpL&4M6PDEX%5C7Jvn76TPn(ESh&0Vc;0w|<oz9wbnHF5zzfuGkBFz}lByr_-^$oaaPpaPdpc-hGu?^5*w}o&7qCOfyHB;Z*cbHfOa=aet0sx-2rXNjk+2Ws@zl6ZwU8-LVp<hC19eg<KDQNd@=$#+2{qIE2(O|r2bE4rW3%7>>U$q?im0wL1Wsl$z#Hy5qTq5hnG+{9(Ul221y=R*lO|N78mJ^32u4BiX-?dMi_k`Mb)u#VT$DNOfQ%U4s=oOkHx)b^t4Y_bxRy{M4(ITAFG2e;!fn*EfiYoNQBoU`dAW+a8VPSXYgYwCjON{Gk$wSge>5(P|cSUP#2Lu(bcG;SbEb*WzLmeyNj|~uAmBNd+6YD8i=VMzEI5yEE!42)8kdb6{m=e)5eb(pjK$|l6d5|WVj0itLR2qbIgtxdk);yMA!KD6li{8wWsPuv~zX|)cO53u5CFb+0l(-9pUf3nFk4zM;hDq|l+|j%a2NTSzWTs;ZWiiJ1rm@gIk~>cp^p)s=O)Y&L4?-K3*GE5iB2!?-;oYY0aP)Q)`!v~CvY<X*pf>i8G2k%l#>7q-2YXljqPxqcuf{=h-wa%k>q78S+)sJ`o2@pzK^;NTlci8Sk_8t36;96=9UqC8SrzKFc!k_V5b<R<Lj7<>ER$4u2T*l)M8{(2t7>to!1bJ|wE`4n+bl=pi?zI;sE<{Xv2d-~1MeaC8rQ`11pb{eC~yekg|f7Y8J+SZ71r@Q0Z@)rj)ouVrEFez%7zcR9CgI8CSKK0lxYVvlH)~PjmLW3XA6~bI}Zy6$dUU^Oy5`z*2Dv*<7gU?d~4+YZ6P@>jz+>VjkYBzk2=uupZ+noYJTHEpFDnPRA(m-e{Cz3^fMoSP5lw*s<<j!i`Y)APziXJv=5-}sjjvTyWL@>=`LN~);k7OPI^q^krjTo=^%LO-vhf2ZE#WOdvd}Xgce%$NIaioKKBE}Fg>*Rf&sJNG!;Ag8I*-;HMiIBP)0qX2YTJN4~+%L7qltX?$_SEJHl^s0oSh-*k-RSl*=VV4apvjkJ=sUf%z?4HHS7|dcc2S6Ni4SQODdQMs;f#D^7VsJnZI-bpqTgVY6JM``a}nUzl00ZG{g*oJVC+DkklLq7;I%4F1LvY%Q360GRGdT$*HkaRPY!1zv$YyunWm%7(XHKA7jG`=k!ml#RK=KmGysm=1(WrS2RXM8yp#05VnWT!mQr$Dbqo(iT<^&h?%v=rELW-~)=2B2gE=ynOxo&ES{I(Qq(&_ndzok9;BEkID7vpYwFSV#|^?E%8@7|Dd0{>+Y~wt=`ve#Nk$)1m@K%sXfeSB;en)Z`Y5u*n=FyG6Tc9$_+qm5*VdwR?7}ta!MAe&qp8Ks|C;C`1)qduHwi5Toc$oUuz{0^$rbJJRTeV3yG^Aa8J2u>286V6#T#2?IvK>Q6{WI@R=rTI{fo<`r+h%0Z>Z=1QY-O00;m807hNR&j+X*PXGYC5di=c0001TaCR{^UuI!*bZ}vGXkT_RE^vA6eS3G?II`#e`4k-G^bXaL6-j=?t+Z!14`-6&bjOo;GWV|9%R^I?&6Y$eA|>0?H}|t|{i*;G04do{_uSpHXLY(Ai3AEjp-?Xr3awV_<LT%Af%n(rv$x)>)4zM;ILgMcHwp8?TZF}xw;p)@B)pi#J5hF>EvMd~H#peo_jY=_?XLGJoX6grYRo!bj_+JQNiV%To`-3XM7esAuI3l<GVeZm^e)Yda5js_-itS9#~;1rV%<Lo=Brte08+5-dl$FrX|z}cdAwZ5%K%;KE^fQtG5xwNuFwb-9VcTi$-Qa3C_Fz*#~uN{&As9ZgP6@O!f4taKJvWpy%!&c-eMWk!z>C5YJL$fcc}Db?45k}qA<FOpL?HPoOzSvhTcB)*5NY2$UT7lnO>ZH-r-{7=iWMwifozW3?>b!?=g*ykQI){H00}~xXM<A7hQ$vC3+fqIsHShE|S8_CfvWPWIT>jbhwCfZe*5SCPgmYP153!hP6!NS&k{l3(QBb9z4yL(NjT>dQA_z(PGi2I^JRyMzL;TcYlw-XVKIf-=^U_iFUrMFe~Wx`y`D+nh@{oDy9ZKG#FpTOD|u|sW(dkvK-UT$ubrU0HCTK7D<*8hL&+2g+S1f#ygK`x-l1%i_3UE|1)Mj^9T!zRUrUr?AO$`m(Z}OY&d()4S#t$eY)l*E(sG)R|_w{;`UR!!47v}Nf4>~Z{GZjworxu5&7vZ=0t(1yq6(TFT*iP&TdhFn~bvAYM$oa^;MRajS$o%rOpCwnx#AcjF*|`|7ZW;JF3jEi)Bb4$3pS#F3-~k>cOigPq-H#DZPa9B3^itZ0X@SEitAv!U=&wr;96e%PZn=-u2!Qf+n*pB+?)VAyJkfEGD!PL>?9`BJTPuxuEyal~I%$X{Cgsb=>k6NgT!b(EIq)CmO_Zj!Bpl^YErb(5K}6-x<>&7Y6lb0&$(1vnW&VgN(URvzU;+7vC(}G9$%x=7mLpIi<lZmf3g}6?xenPuLP`8o#i3y_fG_eDbI~eO+WkceymsoreYS0q-fd`KCLQ`D4ub(ZwvxgB+w4T!ck*MI+-MZ9*;0%jH#}r->ISEt(J$mgb8LQ+pzhy?Ho|!5;|OQMS0<0e$mSdcs*e!K5G(q>vL9u;^vxh={@o#zX7)I?M^id_hw+E{PjB<6hJzit&9#5D7&zN4d}svCzdVUJeNbi=9Cag+L!KKL6X!et$qrIOpX*BQ5}DM#U7esQi=&m^jFcD2aHhM@*>wT&5(O(sOERybNi|2=_@!gUf}4UcG)p%Q~BLQRbWxO;$2yKEy-Z5vI-W^0-hu{5l4fHAf#Og!klP#j_rkQwTnuX7ePRdD&Wsmm8&}Mx0_zy!T=?3klCY69+X+9U(}x5D4UCLcM%WOZLku&K288Gjt*z$77l&nIlLdu&N?l7C`4Z7M3q_HS}VdYQ6l!OIoxf&WN)rW=P9-H3NS^)r(kGea^&8Bf*-a`CjhO;*H|*qw8=s#d6J7mscc2i0wg0JY|VcPPPd+FPjwn3Jc${`F%%Isp9)P=t}O*S5)mHex#|1@IZ*%;ZoPMK2d?N6UE7*=z1^GTat3gY(-ciBG6bMvB<J}L+`Yd=sx-+qz!ffH6VEN;?vpbi?g>!Juj}LY&MxZcM1gWem*;-Qjpd0Rk%*F6|oCxMhTYo^G8k{LfE^{$5czBg!@-66NdrTn*8+Qr_=YxAEnh?R!yyqPsg7=z5DQU@bSYxK6(B)o`gh?-rjSs=N*}RH>dT-+={43I^3=`cyfGv8hkqY@ap8KPkknK4qC3^avsmKrREA15t61uV!2OP;l&c(61QZ!Xi^SDJ5qW|$x2Czho(3g@(8ZbfAl+=g~_}WO_e;6mBr06qb(&9YC=gMSX7jt6F6R*IhGr>Nu<WCRjTX|i-!IqtzFD8ZfX<!sK2vEeEmA7WpI&PlGe~RSz}l-L*b4|o~24yvNpgS848nRjR)o8ij9}N>0&~HGbYrGS*I*m8}kniyq{j0!$8sg?f_3G3;Gv)z1`uFF&U$)P;2Ha;kmd7I8~SBh>V%2If3?|WRHI}>UX@s{=s=iTJJKI`t3HaNd;yp{g`+^oxB%b2f`+v8pX3&sHNhCTI@B6p=J=OsUD(Yizv>IE!NQ{X{!dt3{X(}TIgh!Wke^$rk1NTFC~tUD+zGCCQh9xF%jPsO3*Q_iP*qo<T42%{`n_86Dk}-fhP5s_{S<2#o#I~-xb7(h};)55{*i%Qzw&2Lho2k6(o97ivUY=0dPb@h{C8SNk+-tVhyisa~UsUDDgY#&gGIwev-_JSgU$8F*J$sjmWPR2JAt&F&!(>SoMKa*fiCBAY>8qxfPU>W{d&nF-;^<`h=xY8r>?D|CTTXTP9_V5!1?=DdBvatuT#LH%gez#5ZA7V149@`Vw0BG_T_fX2T{n9K1L?`#Jb@`u^S7QLp2^e0%)jb;INLA3mHO^@RHjhT9Z0KaVmR9Wll)Ue`PMUuPYS{3E8<@FJn9B*TW@er~fUE(fW!IpO3Gt?_K~NUbfbG@*K49=Tp-H4V;Ma+U_TQyENHyI2RI;y?gK;~{m5*lzrYxe&Uh*0-n-G*rSib&^)(bMGRf<*52{O)skjyK$lJRKoL$X)0exB_ha24!<RnbOWEfgD0vs%z@L&CI&MWPR~OxyyMTuAOHQ)GzPt`(n47|%XJb#)lD%PXu|`|lYQtXkQXMWD65m!pnAtTFw}>frJImuDKWZ_UNUEw0TEPQ`mnU1_&_ZMC(>fkkTisKC3l7Tls+x2k`_wM3Ph<Tbk^*VPs<>himn}cm&-&}=vBCgbKxOjj@P7`wLMZjSvQ1j3}&u*9bto*w_aAwi%N76BJ{mdOK5xl!M56eczyc6*Olc@dLRp29-r_w>JdwJ*aoWwv}$laa3_$Izj+Olva}?k&RG@3H18A%$vqx3EBB<|wpy)6%zA@hvMNZi3j!~hFEY||iA`k%Q)&L_k$M&tw}cw~>oU^cuEP9^G%Nk{SE4rkoha=}f6vOQ=t&8cNz(e!P1EjVl}2ctW{mjQn@5j69)ErZZHv&@Vu!?zc2n$t`Ru?|)bI6nTaVrze+06#yo<Npak5NFJ@NJL@FK^*K8-n<(TLmaM<>T0e?ER6eERsxsP`*L6(6uVUd!-^wCHEPjPOsBU!+w*d;5Z9Il9~LjR*a`XXE`zJm?=@?2VsI;{6}?2NwtZANB_E!Qtfa*|R8qHaPfU+`H&SKl~6K9`xh*kZiKu{llpB=;hzvy?;%5=7%%sV4SX#CFwbrapAWd8sM_jy?cz~VwKFsEkeM@<5$P;PR~Bw4PBQ+^{3gQcp6^PI-T$B25DTd_w1>%7;@z|GTUrFdiCc0i@&0sQETVJE}-np!}Se~bYlLvxHf;SXXdZ*7x^`rhx}ziYJ56h$oI)&hXf@lcTsUfE2MS)NLF9);msRr)qfMt^0?z|`=@^(|DP>a^s9#-ZPElD!#E9u!3-6n6Ei}T8e&D-ym<8K?8O;q@VC}7Ub9qg5o=HsO#OtvO%o2iQ4MRq%Fz3nI5Pc))or|1c8KvLUM{nxF3<W3e_Cn^Do7{QSC*K|+v)*Wjsa7gI>rO2ar9Hr(JEzozznJaQ@yZ;@%7Q^#}6-$gR|p*1M&K;Gqw?7I>fL@V)kr|37<2XM-h6&Hc~M5C>)c@;vI*}8Gb?t<=$d7N4e{e`S6Na<ecE-;+C87^Hzln!3Y6LE=frxE|-dm2%``dRRV7ZYED9Sv!cb1B&e?8+MQ&xS#}+d1-2H`WH4mc1nhDdlhB-HN}q!oPo`w8?aM41D+!oEyCdN<CEXw<K}SSGQWZ>Yxg?yiLKnh->v}Im>xt9S?1%`ztL1I!q9gqwNuz{l!W+XaQExfFN=Vd((K5@GG8n~SsxT$!FZGmg1|<V(;}i<&HM)D9t!A(a#L*jN;jBw*_|YShVk}b>4g2i%Cev;ifrHR;^edLyp@u~9K5DIs$<AS`-9<}&Tk2eeR7Z)BF8v~==@XzX*NVyd>0ZS*<K!~V3sogyDee8~$g^1OupB*^b|mvZk^(}4<``?lZz=6D&sIws3fl9&%e51zvI4PQ<7-w5;+FRv8j?nf@Uq^|)-P#G80?6;OpCx%KZVV~XUPIB2;dOI+6d0u%_AEP`v>RT5_4r&5LFUD^X;JJJwdOedsG<CMiO?p&#wvXH)LM~+4Kl2S4zn+xl5~pbTLS*fb6n3Wy?VcVV>C}1v$&m;m-gnyZECf;e?(U1%1faBS$2v9(Fu5Bbd*RfGM(YwSdo$gnv*kmSGT0X5nRiB<Sln<SZX4I2%jJQN{lWs33-rz;5H)BZUb<{zaC@N6gWr`^y-dz&q&>kCw?Kennrvb;C>g&i!tc0O^SW*5XlvK`u$f9s34iNRux}UJTk_)f!pmc`(1w%cEG{Rx}E4JrKt}Sge@VQ@CoM!nk<~+k2I_+CvKiME9uCWkVc}pDVxhdH(U!r=eTY7$J!oUuuC}6P@+X+uqZsUcWck6R*Wh1k3f9|Dg$~AVKdok%fs;11pxvTo3Cq>PEyS3FV43tZ|Cc*nRc-{d>PfODF2)S@(c=z??V70*2IU-={0mbU4OZ!gkrrpLe`z$Gap$Uy71K%T|eOweWt?*jtCQ70Hb}VW0g|FNGrlo^9DLNZBWFfiEGrzvi~jcCdX$G9g)uwWfD&mQb6~+6|TkFDAH9(yBM&RXjq!$g&qNjuedNi{jSrx4Ww}AqtoWey`ix?|8dCx1wNVo_4%B+7Tc(U6ORfI6Bqu@WxkVWqXW7ES}A8NOkG<K$55eIQU6Z@AU`Ix;?IO70xE8Pz?gqveO?x>A@c!QrqpC_Dj*~?sfMuL~fcvE)p^?Otds(<ww-y3b$qHYglGJ0Cc2h-+O+a+Hku!Aavc?weP)tcM$J91FmiaKer<qQSY4Eo2Tl*PQT;r^!d?^N$Iy;vRda0U(ny~LZwVDSJ^7}wfMuP*0_lPXUQ~X0>ZAK_|k^nJj>xEC=$|ZgdRKacai!q3zwI?{g|6%^`@IAvz31j>KfqBUFAfsB$xM`nO@&ShB-R~gYZc$?)89I7cpY%pL{Nn@hV*9q}`l6{Vdy)7kM^Y!8w-Eas>DHFe%imIEZEAvg=Ksu7}>}<SjNX79H>H>+|Q{luY~+W0S2T1>U~qLLxeSc7l}N6eaUj%)s?k5?v|p1Fh^Nm2FclP1dkRe-PxetjNg}hPFC|S+T~S6n~}#^dA%h4Fn|ok|GFvFl|QWng&B(3KnVxQ$~R<z2|?|>fcoTJB1U>s9ZHba+$1h1)g-Zu>q0>P1OWIm%SFm+RL{jD^TA1@dyQN=^*o3K1$>3K!Aa9pcZ44X{#($Eoxz5R=d6Pj)u~)do(z2Klj#l-2f}ZpBk`7!;Uu;l$@7iG9MC*rZUz;bLbkayt!Vm;IdJ`DBn6BXBg!G+YTJ=LPPi67gjZ6+KUY|qDZ1#RDXM)q~)>-HEFAD$CBdNDTd&R<-`chFt?Gmka>~@lQ@JQD92B-t*YN!sJ=$#=(_#LfJ`sJ#i!;oEy2q2X47Q_^C4;&9m2574&4|Xx~-N7%J?@`Lz>ltUT(T$=ri8YkvH&2bB##?zz*=gzflx0Br(yccTP)#MB<G6G*^ui(CwZSV7TAnCR@E$)#EfDwG3~IxC~mLtJXNl+4b9U#vl!f?%ojKMCSE;&SgGv0MZe+!QWq<{=FrXc`UZLyyKqoJFn^U*!xMtM;Mo1wpQAZi4MCGbQ6pI8?kC(*Q$-z(AV+hZ@jtiN&h_gZ2!iO;PtzoR6qYp)&KSu3hc>S5mNJ9L8Na1qEJ-HPk&>MMg~;=Wm))dUrS$Jy*Ley|LyGK3q@C``O}Mk3w}QSN9!Ca%^?@Vqm$!*|KyZ9{rKUh(=#c!rS>eo45QmXd9euwW4o(<n?)v*w0w;p{O%twem>(}@4NkRD5Xj1PfSV`gwpSP2Mj5<35wbZzZ;Ay*Z`;6+`b!1RThAHYK8O#khSA3@`}_4OvGJ%Gb#e^9>FDjq^B5<3>1FzyE|C<*zFGLwPCpF)y9<$@=<$X!hYFKxnE<|{7|>M2!vVq?JoRMOW)`}msOY{#rek{$*O3B5lLI5@AxfRlz>B>W-ezSkPDgUz{0z`gDBN7<Lm$(iR<wEZYbDCY=Cn1+E{*PA5M;c7EU0&eDmh5@tU-4{la3Q+!O8te-o+k*xUKtA7hsjTbrK0?yI1|m^TGU{K9?HF2xj=A1fhs74TM<ip-&Zv<GA!=Jb@qZwxs`Qv0Rek$eeNMD4ycDAZB68$Efs({HN=dAB)}c1FEnbj};<67s+xWka=Fw#99;(`Z<g`g6`oJa2xI-h@4<!W&Vqsjap*NtSsbJEtX5)b_ZC$i^K+NqCT`ZZqtX8CH>Wtg^)%%ayGp{8hJ0wgzHpQy5;QwvNwBIOYVOr2<#J{Gv60FTN_%7SaAv1kzt2liFQcvvb0uj|dEQ`tP<)fj3e`U$QY$X|8FhUo=B^K(lkGCQTW*19@FSUf&;bg<D<uI!VXbb@_aj(Tg5<!Zi#%iwQ*7hju&H^m*)&$~jBoyiijnyXLpvmlgIz=HV?O=uCB-r`lG)xn@_{HxUWWm+gl4Yy?C@evc(QO`oebYv;{6ZKw_$atZ4lgpdJ#mMJLDz0c*Fwe_Z~4tx!3z{HLq+Vp(~F6l7SwB7Og%!ecIFPH&F`qb|4=O!>~1I)S^j4|x@1mdL85PS&<)#v5u%e<55dk@p_nugbRe241<atS8WJ!qlv^H_&dL}L9DeGJVQKkSHI%;GIrC2kgcDNz!V&3{Q$((xJy1x2TB(OK45R)!`e;c>xQdf)NJa8Vw~<BKet)hb~g-qZ4UIRvRyeA;*5^Sb&At+2W`7ES-RBkAo(Jl+J1%KAK3w1B7Qya+2}NC(ZBaU1>@;sD*!5#qBGYXFyGPAf3}C-DRJ^N|9qro-S~?=8weuQuAqW#jUtJLhl+?3&N~|D>AQ3EA37(n-rO;X=2=3V;d@rw$V>e|Gv?qoEBfSD9^L$^3*}_rbBd)C%Yy^ZHsDh(d$HL9t~K&M(Fx&7?yHFB`~!aGcD1!_Z_P!6aK=hf6KUZ%h#m6a|yCF2Ap+X|hlJ8}0h(+vu`&gDoJ-TJRFy7@k-$P5rQD;NIw$FfV+o69a8U{~`L29B-LnCwJ&+)90x;*`C51TH;Z?Y_QjFXg2*R&F7j_*CWS{D$7^v12UmkXL`yvMk6oHy-H?ZIIbVV%?M|e^ICbUE(~|og}9wqAoWnA`n~Q|#R|nn%>_rwFHQAQjiiMs3E+ESci{uvRfgwtqeDviXE2Jsr6XwaAdH`UrU4Sp>Hi!KNQ2~gQUlDPL+{3<DM)v$4w1+mYe7noMO(c!JUSp%wHS<sCRo8+QE1|tN)tls$W+~}4Dhn0s-am3j9--u1w8h3xx3(we4@NdsHWCdz$a1oNm~@RAk6(Mg;~oTz!8v@V?f^VsyQ;yZ@g1Z19lhuam!alW2p5jXV2Ut2oj!GPSbQvt0OOR(O2@CiyAKfD1zlF1soMH()u4)b0sOg77uQH2376Nt0FR7;_c?DdYRA<_S)j%sC;nOsu`hpZG5Ds;l+B>3+iy~jj_}aUR0s-A$SBiq{g-ZqS)D>SEQYca|2w3Hf;Q7OFzLeLR0)9q@^}y?a5s%vtRu;-7-|(*EG*h%0;3`{UXeTfT>Y&GBRz;%l{$Gk^UeSb1>(?2!s2Se1}7+gJi7FvKZr;&Kuambl$`hsHmbS=)hpM>ST=iMVcu`Sx2#&2;K`g`P^<~A?4UM_0f{nwOJYI@&=7Y+5|z%-AcQk(hRMcIkU`j7hc4Qyw%)v-qpOAxUf;5)e3Bdnof6<^c-O3^uzbf^Ka=9*>bL@`>q*P+@E7kw2;v4G8sla+wJjUu|Nn^bmvB5)a!O<aSCE?@ARLmZs*>FgWuJm2Hp@sYyhUgaw0U&5WyNny`eEb52m#Z0)6{*>T%e->#kYtxzk#o+B)F6@df<mMMwK%QtzLM5O`M~E3K_SvxklvP*_SpuTlR0Ye+3$O(w~WqmXDs5WrKL&`7=wNf%dyR>uFcpA9IfDDAY{Esq<*t2<w4PY5=%Pt~!=D4Wld0_PVh$5y5eVmS`1{~Dq?rDc6nI>_2h3~dW_owAK(5Z5=7$z3FJ>Qaaiv}N@4rjmBDF^i_kIbIlzX6aVsn&_-ac~XowSqhzvHUzK2P0BiN45s3|u_xEWYrxS76)w;3N%3K^f!q<NK|I%{_zyoHD>_$|_)LjD&@nXJH|`36<2P-4K5Wtghpn>8gWInB%f)qOxH6V}3}+E0%O;R1;(F4ZzPC!pYH5AqvmeQlZQ!o=Q89EM&w?v&L>0#8klc3ovm&2am=`ET(zWXye_18#khotN`Bff^!q%}!2VY^SBmV3G%<B5vwzYIqwo<E?cM5WxWx-hr??(iC_>9QPZyTn>^e|-7=inV0RPrfm5>TqyQDn-fo5i-Gskh?|s5pD8%Jxcvti0-I@FuEz)T^*J_Ep}{5cqh0uL<N$WVz{Y>frusm|seBIe3^j<(+_Zn>0=TMPk(S78RDSNPl=&qrjfVeRnfzAf^Ih>Vnw1v7kR|sl5KC%9igx-Sl4h{@FMa%j{)z?zsQ?<=TBI){qPJTDt7)h?KL%J6J~?EQvLVMbbpiQLW~-Jr#j!xpei%1YM^9e70^aq}W=(_m^1u*cy1geG>r{)>ehaMs3eK!*QMPX({{EGe#Dkw>QI_Zh;AJ{eyvS*zPbtrn?`zkoWWV>kp$Zh_V^Smr!qJ*``)TM1Alalr=z>maE@<^`pG|BDKWpUw$5&Qg>r}G=~pp&+`>IH+>#g5_yPI@d=+DzKAE-(`E&iUT^P6)fn%OK&5YNYWv6N2=))<AT2-QK($&I>Ks>p>KlA%<K|x4Og5V~pHf<48+V%;*qFkRJ2fT&sOfKH4jVR-DyY1rB>P8qZz<Kg_dEBNHp6@9c9H&k%eK)yc7fbxY8xkoy=`KZllRunNg?|tx|oXNxuK;%i0;!2G$=v(B?+m@F5$E9byF7rZELq|N~{u}ijSHbVvUJU)<eVMiiXdP1)%aItH+i*h8KwO+bmcaS>T4{537mF`anz7uRn5+2}dB4^pYWRG`*&o2~kz1<O)yjE|D!tRE8o>q6TFKHAP+>ora-99A|uMXk%<Zq;{AYibw>LO3NT-24chegF`E(3?G2TK7zC`+L6LCbqYMs+sF>wCIuH&8hwt^Yl9{*twPOJ=a8|YV8cp{_ZA(*aS-Yt{w?d5DJSYc^eY2IPl4!}AgWXqbr|KlyMT!A4><!Wmb6L}s_a{?YT%{8SKU$Q2?WHE?N%bo&tOcPt|OoYh7(IYagLkl2TK2R1T;B;5dy_L4#AK90?xzVkfd&*zJHJJwJz{KJGS?IRL;Q@1DD^EiO7C`cKh?Fx<v+~#&4h77+|Cy>=C(V+@mhPbfXB=xDkmJs9{T2>+UxpF@7w;=Wm68eKc?T5UK=Vipcdz4wY34iP07_WFkUc7~jZ8u#_ZXZ8;Q6eaPfoMu$BRsXnX0fU85em9d44TYgFcK@zg;$Z1va7zH|AOOWCvJAlW0($-48A_-u0r)wEtVB?qVHAKBj*Bsl8LqGj?w=4^mQM7p;3(0`TwH%X@I6bH00fGGX)oymN%CF2O-4;<fWEZ(K#%1gV(0Zy;3YC5(X|S6VF^QZ86BG<8mYvZybc(cD6OlYcWEHEt7Jji>%;dnzXeUL`d>gV1+d<uMJXSBTDQAuHf+i-`J}Y;O((xm!-(W6|%0)d^L)^4wr~wg<-B-X-0^}%-t{nOfdk)+fsrw8+875Q{ns*}-jF)v)sv7BtO<exoBn_kkz=nx<*7vE;XxBO^Aqzy-g~Ik-Qc%ewV`Ho=sklP(-k9uyH?GJHlGpO7DVYm%{LmG>X{I0wydY2h8IxSu)8g0S<|)}f1H8bTss?e&jZt(T2`jW&N%5{C*3@c79y;fB$Cay0@p}3n#|EWzyDrE&D$T)>;@R@cVHhSmhkC8p`OQWRdm~>zl?0w9Lznan$dDKiGm)%a#L^AcDCxT{NlyhY1zR7MIj5KpHze9(=C7D+s5yJt>}?XyME<LcPj}`6tK|YirXlq{$#q3J4G2kXa2q~*KZ_>?lz;VeiBxbL$Zx!-*XlV9VXQI;;&_6VON0mIR4tDaqrBxG=JH3LI%|#3*~fpU2YZKiiI<L`P90irVEdRJkk|CN+L|9jfQ-)7Pi#)ZGiCe?xUWYv&FbL-=?=10Za$L<JCfEPB#fj3x-ZW?>7eebTl6lmrH}D(Kbn|%GIIm0>`5LAcDiaxS>MU0$s))Xag>miBhz@~Y%t-|>|0Px+Av2Yc8BNphHrP$nV%=zkX0V@qzmdfvkSmkuON=V=}@XIhK71<rVZe?VtN>(8C6qp|Dm{aq*|+sX2PA}yq8=pYBw4wU`vwyiy9}blZL1tHk#dfSY&m-ESB(=*&<L6n=oG7b%+G#fius#RmESxb{IWC3GjoLU0NC{#ltVF$z(EawafRyWy~xPx=6E@;?ma<x3Ik^d;u%FjFKXYsM04SR}gHe2`3f<_1N}7l19sT9;Zb(Q>Dv<wY;!*STd9n19{bW2X4-J(er4DOj44YQqokal#1ir<g(Sl_P^@W>^?F~q$`>Xr1SOr&#HNVCr}?7`uyS5i<d1!2g~Cx%tqPL!c?nlExwV)=m`?N;N1Og8=S5^8t(Oa=amgFoj*cCr(agRUOZW2Z*;=nfAD@;4SM}O`sZLj-lxCM2GK98!|;drm(_0Xfd57Z_#2M#_u>bBa_If?_g=5ppB(HB^tZzvLFpar?c?umAAbj9{(H!OpW*f1E*jq3>+$zP{yX5m`}}<HOZrOzMTN7ElJqdr2mu+IPrC?_L4NP`2PP8i_<*2NsdsB4tO9K(cpzowGwH3ny<W%8@6av{;Hnie49mGU06=dfAOpfv`wmzF=dq{OF0#zt7DShbgH)2f86TOkt8)>KMb;|SfWIp(guYuyAs}LgeQ^T$-~bhk%s1peyE2}87uL(Vg;yw;4Pr~WM9aS4#EQZ$i}iv}-d*S)&71_oticf&S^@|V^{Wo05rRV06I2EBaTV9!S{d@7F_N*@#@V{04m)33l2cgHmWl25C2w;AH4GiueE17JnJ4)?<OHY!5J~YEQc#wa-^vbWo+Zvs!#Xc{pVo#FGwhXJ{#89O&yulaFXYW;YHlb@aY}a0(iYAH<^Hlp+3-eJVEAB7^cpr6jAA2dhcg+}EQo6O<tfJi`O$sOC=Ru75lngc1q=DLsQZi7eOhE?{DsP<8(oo97&^INRsI4_t*fVyZ%Tmb!iv2q1^CJkNhh0kBQmFA-@-Zf3KU}x9@R!%_ecINIdzD$artv={X&z$5CKIl9pGQ8V!E1F$<Xv?T3l6DV78e0DBAI+M_jetm2@z0ifrr^fv9z(N6#6*^si&bBEdcsc<xe`0%1PFa`}*nU!xHe*Iln;2OoF|J1VKF9ttd39!kxh3V1MW<2;h%r;azaY?{quZrugJO*I4ttmevx+NH1g3W=lm?i6}}Uu{X-6M*}}_1~a$FS7-!Hgg&&qpz_DZ?7E5S2(e1)MORroI<?*YVUa#X`mw%JhvxNttk|#mK8`nsBTE`r$j1hZQ<sHd#tcCogGkLs355e<vN8jA3UF~wQY3cx&&6yy4>PWWe`yt!_};oZBid}j;(B>R$mpY1*1@fXtA97GcuIOTK;oKcdK0%k=)(ZDQe~na#;cK!ekuR0<fe_MZ#6Tu9mKWxS0t^VGJ14sn)(bo6hGkq*0w?G_6vsV#kGAvXURA-Dd8Ak8lv~o78~U(6LudXb1cIy9b-VOsB$yY=*tlZ_~JIOsLroHq*h?afoKno2z9RYOf4U+1kBH=CoAlHR7zUEC#LWGzWMoN2JQ8cD3BVt2T8`oD(u^*qDIO#2w8#kG)V+J1sxt^|E=Y&9F;C+`7S}hO&iy<FlFL-ixtv(wuR<C;!jzRZazd<ozegHI2W+(*g4wCC_D}OTIVRv{K`q;`OnYuA*5?x^5W+?lMrtG}+l}xFzG1?{z78TqO5D5wLJ6vF0ipA!`;%Z2NfTDm*RiaPmgy81$OtMAGrnkgX4AmCt#F*On8Nz5fDQA`2pDd|eSqS1up$K`pW{Wd$(N*zGZ{j&R!qz$hYVZ&*tRlBV*dR&vxB3bU!FzbgA4AZn5B1o~I(=zs^zA$C;1*P~z0xuNPeik-7)LX?>AOW#LBycnGGHZG4LyvTiM)_d{J0sTsEY3vWct8%L(0vKtgsB!h(RTd=7hml<A47hhm+?&^CemJOr<pfo8X2@+gW%Pg}*_JcOBa{5?-6WsDIgOoos=)}6x<A&YrD>3QIusWpT92;OK1kFtOn2XJH}7k6f<tPrBc3nedTk3n*2S-+dX^Nk({O?bod8StRf3<vCQXkdOQ~O-a1qJ5t<1bxcDdNRQ&t8*IdLYar5WAEJ+HFzy|?Hp=hrfh#|XY)Dy_E3<atY<plDabUX>~0Ip2Y_TE<Rf22XB9D^)qQIoFSO_B(gLGb2vSkQ0dDqw-1+9E+AA@C>M@Pt7*lW)N5jXbNjjWg0qZv9DYG;>79a4MFWtq8gOK>2$N+A!;}N^2xqt20Q3)<gof^q3{sa;vfZo6ornr$h$Z^+{81C?z*oUK99Yh!=KTdT;!#0Y8$%hG!$@Ar1b?cRLsN%x4ID@%=Bo~qB|yIt(p!8I4Y^|#J|dFbv3d9h+!csq)v|p7Q2GPAN?jyoz~EHz`)ysYVh9(1ILpDiR~o_<ExxE>4>8YvG}?Vr6*8#rgAe{VOc_?jr>m78|p&dFdVbr95QO}Ve``#ADZAzYr}U8!=oNa#m_fIPl={duf)$cg?=^6G%Frz4e8qk&K-9KtjeQrvs2yW!LT+Gnhb2p9#j@_D9-0``AbvagpQE3=R-^_bxEOVkYK0Fo-E};y1#R)CQaEwVExCHrRwn3gN5lf;<Ac3S0b`^27FXxXTQzSKAb%oAQ*-U-Be;u%P#r#2Zn8C)m3Z^lsBUw7rpXOxH@GO6dAv8#4sK;B!p<lsDU<uEG@+zlk}CTNuIs7(fm(KigX!(Tovb6$J(jg>N>XXm5(-w$AM(sequbiNW#2n9{<@Ts6mq;Gmn#7Bef^V#*m&Y8(ZA_YClc@opktSH%akVbI0Q1yB&#*;@gMx{#<z{s=70-yzXvT-E`OSw%#6RI%*J~3m3Zh@_BNqI4r`kS33p9h;KRvhSlVh1?H<chYTc$jV^ZdW7ML)=0qi&S2fLGY>mCiYBt+3ykLAWO7n`GWM+S;Es43r?uR;8!o<=)P+d{^W&d;}u}FBLkFM~;eJ*(R+W3J%qFj;(W|r-{2IJmfsMu@hRY1J5bED64;h-nsN%tCT#mZRC@X<%LGCIa8xW2QeB~4u!Qi!iSv4?PXukNw^J1|bbt>cKdc+IToXRqz;K~8Ef+jSg)>yzkY;k%DWZ+zzmyChK0Ex@DQ^U-iPU{`OqXTRFx<7(=g&f31nuKj+m>zo-rdu`kiu(Qi4@6G->qj^HZS?X!9va<6S_&VT{1ymiD>ra4bOW6-d{sE-{5=|a?{rHFS9iUyU0S@gtN?mzR2<hz;3B@i`lveWx(AAxnYd*SiZj+9AK9nK%?*aKTgJN=GF>oR!;-hQ3U@^D{G>ruIdEt8py}$(@INz@@BQNQD2SDkiV**t8eIg2=51_QhsjqF<O9GPat`^Xz{A_w;%*_!;nDd<``4P7Cfcv&$>PjjI!{xUXukY~jlq0lUu^cE7HO+~qhROzIvsDK=I(2O@&jYMjxmWQ>-G9P}`ktr+RNA%*L$s*dV9L_KTv@uoMyGXMN;9Q{8Uk(V#FsQ~a_IY#w0bHx6q^?%JG%>fQ~Dyt1v;|1jQmgLozoADHrX}kU~9BVL?T~B3rGk7!uP~D8`lB{U=TAe>U9qk{~2`m9X_irxwUU5`CMaey9RHns*M^WG4^V~6X&33!*jFOt2OY$y}>`LLqD=3S#3|Jb|0vbild;}L%>b{AGF_@5vMEH;Wr7yn7zy7)xDtBeN*Hr2wi=XNNIn!)t+>{XhJBDbXjenH+7lFP)E41{W$26TE!S{h&ibnL`<K)h)S_Lg@BrW_qspe_r8{_^m`u<<o%$mQ2y?94>vHcrjPHmQ~l&t20i0F7wFsl+~qg;v)eekym%YqYW^@QS&@LYl$)(&?o1P@;iI`pV?XDT^xrrmE%^?)lN)=2otdqjQDBvwzHygl^ATPE7^$YtD@aWpa`jPE?L5GO-yG^xt^<(&hRS8##}bmh46ju}?O}PCOGUDTUXDAj%6<9D>EJo11V%)lpt=G7?U(<U{`qlUtM!bv#6E}dnT`o6BkF-A6|y;eM8Pt>l$X9!#S&=yhRD)~^o(OPPORS^!tNa1P&;wdKE-ndXH;<VJvraTfl}-fdt%V4)#cpkvN1OMJR&5i0hB%elA`3Phu&2OeyBf&K1@9fIBFdn9s4>H((1HBQC*I-%autVTxd@sTI^VdtnI)x79XLg7P<-ku9l_*u`xSo4TY_AaAhUG4K8jC7DDgUbH2_s!F3=U$Tk`toYy&Hi<>hDkqVDQ6Ps|3a~=xDfimYtcpIeBCcW!!1?aFA8fj}XxV_D!7dHpBqrbTN6MOU+#*f~Cc=Ir;wi@AzQK|5qx4U6G5smt5Qe6()T|p73<<RziQkg-&q4~-Kr?neeh6g>CDeX<B8@I|k{d{`IQBBr-qm-+aDATNp2w2x50&qwjhI2!BfTPn`T7~q4;OmgI;j}@a;GUw8(kQr8vmRlKoaR_g+_4yr%UDdk2KIVi|3LR+!3lVZLw##f&X;u?q^|EVJ?Y?<BofZyZMHIlVA<Hq#$B;6Y~Sv6_e)NTgfD<B{Xb+`2Evzva570G-{iIHCHjJ1IZoe(yGvJTzFI6Knm~G>FC&zj3BQ9D{9TUwN`0d@9o?Woau;eKa8eq>ScH5NI?U%U_)cJ`YdB`N1RRAAcP>x5f0GNSe5tPs>U}}#i}**`>5N5zCr~qjMTVbv%no^+ptHU67u|7Ogwd7XZgguHhZ^O2m5Nv8m_|;&yQDP*2g<c=Gx7;Ebt#K^gR^NM>Z_b_HzzM#Tp6b>!p*DXk{sqyUJLDo%gYrGJ3P=_eD=)t0=U9;mCDpqE&1P9*IKtn`@?gq{-{NLSl$YfbN06V)`~7>VJ?<?aDfnLEIQaU3}3buTJqX{!^c-!pJp+c67kIfCjF}4v&#%N`4r5L0pqX^V)wjVjag;Yuv~N(9dDzP?m~QYobHOWB@!q#-(5duH`+7HV>b)aSYLNwNhpq?<+@~QA3FTanu&YdC=OutJH=hb43)#=v?pD@(MnpmDM`7bQ-i7BGSVBCrnZ0I9VEI@=hMqLoI2D)4Je{vUM=;FeLA^CJmBa0y}`WM2ZgRw^O?`x*;LueE;u989h0i`l=sFYldjv}S9i5T??owG3<?Dpiojt18Gi$+_jjMUwueg`2NqRS=PBl+;c(a40fq2at~A^o`j@8oy<*hucGYd8mMg<8s6LN9NtZ(s>ee`Jl1*w!<hkbx_qa2NY?1`jRV-sZ-9PfL1A2}vaV~ou7-Y;(V(}sIbt2_ReW6U*KdBE`w^N<J;8;vq`+S~3b&0PUQ96-}ID8t-)Q7GoX$kw;G^z?2K!J&pd#y3$HY(Q{?u#{jNyhTZ=M9TX&5^&U8+EI<X~n4q%~i*M?M2581b`1Fs>(PEk2+~AyJWTs8hIS9uVoJFJ1q9io3{<?vE$uUtX<>7v=k5OY<Q2iO|0kzbAm3YIb7`ZBQ57=2ZwTLje6LI@;+b^JikA)*87D#V5RpOge!+7{*^>4D~mamDFOJH#69Q}Q>a=AiC+?Zs>^{xOwEO07;k_99%U1inuKs~Il#35(xS@>R0DNd1su|p&Y40UI-1zqw!R@(iPhzyRH#IvJYbZl{iN!lD;HfE$ED1HmEGAc%9`fAzE<zSx2O&~(k4=!sjqy|cU~~rG;3H~e&CdMRE)4F(i)=8&{)0uP~Vv_O&tZsJ#GYFc6nju6YtR8S}?9@kSvU5KIG@#dfo6^=Tde&`RQeG#Yf~Md=xoVI8sa82w*$=hi!O|jHmRdQ}&LqvdaZV`|y+P_n|NLS!o!^-#z}jUsXXUY4z*8!41xOiRxT~A<TDXKtz;magQAs{(GFp+w)4zX0U&t6qkMOB~K*$2Tg8IsdBK#hg>)LFm==F%M+?_&;SmDcnNq?t{E8TIjy@nh|r$h!_A>TRboXEB`<4AePgwg$f*Y+c`sySR(|e%zGunCX3A?pX4#O&`^0+>{%{hx*MWd)M{cpt_1v)dgaPjPjyRLtNmnj<((L04&cJjC7(j71aZm0`#DbYOmTo{L7S5M%P5+Az^{P5vq7y(;U#tN?AO)lhw+0f;I9VK{-<`%0pDi0O_%v;uVIp`smUY(_&lL54)0O}}=C;6lzBX)@Iiz0kTIKh3Bu{xorYU?>r3N>!IC+Ej!OLKlX<zT*xuN7OW(ipA$P_l`4Q!hSY+QBAk#S+pVN@r4in<8O&IJtCO)0^f@^}$&_xy-M);K30{mijb-hezAf-PUlcVK@VVLv6b|IdN`NDlP>83ph5q<Jz{3GjqcH?Q)C^K60M+&Vu03SPt`<E2$&No9vN#z;#=>#b~0!X^Y5!{=h_SO*89L9w5uKX}HePx0pmXfOL5QK<BoJrr>?t!K}kHK<#6IuEygHt1@VVq@xFHV+bY-eG%|H^eG_f$&;|CzBEKvySaXaS|n=w47;FazOyu;FC*d0;Kzd)kCo#;o&-fEx8LX*WRAv8YA2)ov>0RU5)l$7|`8yujtW8ke-?-peKX?!{}{9K;J|FRk1B#TR<uqBw|u@)1Z&|=RFQ@gRNQivPG~i$$XPy<trSgD@E#n3}(sg2*k6UKtHHr`~E>CS{^e95y$d<G~7F{#5Dm`FX<azeKiI01Eua=Go*)F;akPC|FM{~-Sc|Sqf$J2+m(*$e){c-hro4lP=$cvj}uICg3#>-sjJM$nuLBxamSYJN&;PO(G|)pl~n<5!$@<A58us)Oo{3u0>z^nJodj^)A<H1hcz4?m#tVRjplRxO}*9MrD^=pTE>5shVjQ~3p%UoLzRR_TiJ!`it;?xSUDEf=zS$dN<nY9bmZo7K9a!n+vlOY_rCMQAM1z|gFjdj&$iRTl-ZG777jkdPB^nB@rG%9tCD)t+schovSG6_T0k6(o!^%_?(m<#9C>^7MYjQMZjbir(?;#?vDn74HpO=7?s*V$g*o`|`2zDzC>8|5k4fDdFe`5vnBIHNZxxo@LMQ-4H93|vfM&hZ#0HD<y~bh<IB!Dv15w}ie*DqfZQ40dA6})7gRnL!qzN6cqQpw8XjRZdp;_%6ivA?JEvr~P(X*6Vi%fD_t!=p4XxpdK({SH7=3V|dpmw2|s7ItiW8$f)L+n={qGyNH=Vu4c@aGVJen6PtexLsi`0uWv-aV*SG*j$vdXGOPh|MsFG%g%kPdc9ZMT3iR(r$_1JeiC}+D^7nwf=y9ZB@je?~CWP_klFXrC7<6_N|3bhX+`VOTj>pZgaaSR$Cg1TpgY_B6TOq(y=tJPDNCD4WA;ZU7O`9#!5#5gs{q^yj8PoR;%v0^6ar~(KC&X-No5WrZjZ12NvC+in^t|JtIzrkK}w)$VkPc&8<o&R>a&l3KCMO#f>0Tgk6<tKg(5ct&OX)A`A5UFV;Ku!!+0j?28e`jFlcPFXdcCeQ&S6zg7Axja{=%JA?Py_G;W%Z{7ymxZ`DXfLp5ttYL^9=Zq0T!zGP?9VHfe1p>w5<qqL1LgZaIn<8x#=kSpeu#%gRj~5N?ldsj)o3re4N6xJHOcvR-hfME?Co8gS+BQSQ<W8X9zCQPgxg7+SjPpRe4(yC^azv_>tW}CyNy%E)zP?J3gk91m2W<QPNMfXxzFRe+zdh04%x=g{l2s|>eLe4#zxH~=X9hRrr0GTDwWZ&1c<@T2)vJ<W^95726kEldxY9CE?*kRk<To@RXZqwwn%x-3M_iphyqjfI4KT$gZJO_RRJTFK3%@9DgM3cyYIojM<Csw49s_zxs!PMDa5og72(>bhqP&;yDz6y`y8P5AN+cdl{ZS<Ms|}-WdT!iRBlp_VKy=l0Q`PmBst?CFH-(e%+?3osVIP<GuWG5FfKNUw`YqI9)1h$(ahaU_32BpdgZj1rwTMybc^T?>&bP~1JMP@cKwnK+&$oA9T1+~ez)`&4m4XKIWWXkF`FFr>jE3Z}eC2M(Q<v&$(lzcIc5fPPGA)t~k@xmE?-~(4e0{r4#LRsvZFD)wrx2^lNdFr$-<30>zdeyfOH(n#bUaa*Q?v8rNd2x665|3RsHiE>*MBxskgj!~^(>&UsGl`J+IR`ECV<FTUJc<*5nq|gSD(^Xx&72Vc3>cr=ia(0&YgWpz-?TAtYNlCJN@B#!`cNR8d;KDhu6fEP~}^>66*dww)8FQpPMFaEs#s&%>i%0Da%|30dluJ*v0^v1O|tiZijo{Ag=&OAa1xmympzil*Y*bHQEgWbw_C3ba>Nz%^`Ptt`HUG`8ToWZynNTs4lI#*T90o!-l5RP(C8@+jOX`zjYbmOiDE8Rw(BBW-jAImYiBqz0J(aSEfmMm1IEQD~m{7K|ReI@1Lfkv=aD7>)cJxzl6_snvYs?G)B^^>6`2EJTJ&5YuVI>E2;U4%qjm}^I!O~3EZglrG>o;e4Ccv*5$XX4lBN)+2*r`TtfYH%5k?EA|B=1PENgr?$URsA}7t&RvOUMk_t`NXpC`cNJA{&DkK4ojaU-=+GvvNaZ)Cgj!ijwoa$|b8TH<U67bdO+zK>mr*%tRP#TfsM$+~weKYcXakiUt<6O;Zak&h~HPdJrM{L|9#Xp;|L+|79=XVHJLJl@gxt8&<OZx6}7Jgc{!K#RmhaXFrB<O^l9j)*FedoLR&UfRp@7@l-`)T;yr~iq=<|uPH7k5Z~W3I&<n{)bua|Z!gV(3Q@ps6cdEtr>Yal)1Koj^V3?m*sRR|_ba$3-Zq()ejZYfCVoi$E?!h6jRwGaww^QYXi9ozjRNSNV;Q!w4t1=r?AgvgUM(wbgtcE^oyV6j#9;g|fq%w!4g(w`Q1gJVgc}l1h-#bgPqXHJ8l^7dbJfRq`eVe27ZPM(YgdNezg~v#9+aE{G9uVJ8)}D^>wXFMc|GfBezSe-@Q#=!{;~lY*SsOxjIgXZ6p4=XSAJ)-wE2C3VkD)rZ=}H?~1u!HaJq9QQuve{fbW%<=3#ZT!|uNv-elp+4+~!+k!1!<Cb1NUID-Kj>$T(ge7J3cJOI$$|@qr{~?zXYMgY0FW+mGAHABWsb}#IDb0ErJEm5KlcxUH!nV&oxV7G%aD$GjVYbX74X>Wchx1JWm071LR!s2Z=7-B$|RRHj?}{!IJV|ptdiMSpNx#XFe;KYACQ!^_y4_we)oyIAB>}s@DIUdQ5o!zm{4oKIECasHsdg9S=ZV&;tZ!=?Ew8cTqe}@@^lVNu_k9N4=_-ref6)C3Tg0=5te-#DPX+HR&&4FffkQrIUGJFqsH7tLQ#5PIu`wK46Y1lGrx};&2V%>oUsGr)KPi-+dQEfvlWO9TJ99mJ>JTFh~U&!!6s~=a@*!<=1EyyP9KT9%5qL6ifXjPLZtY_`OP4k1baB2$jw@z8-gfQ;LL4%ZqCMr<GX2Q&|;mQamW@KlT<oZRv0`vSF0B~H)`gZ7<82fQkTO4T)99oB-&!H8<OJ)Fq*N|Ean_Xm@~rsVpUYf%qknfku}U2Vs*2T8N^wZfSy2h$J#fJP2(W@99~-LN*)K?SmP~*X_n41q8{Cj^>R9M7n6B1+b&9NsCbF#47$}M=B&ZFn}fWfPd<w>hOYpXi7F9L!H;&UN4#}qjzUG`Tf}WbFt;8mIQH&CuluYqmyAO7rdyh7Sx(d<qyg*o*X&oZ0|YTMZax8KzSCJXsbWt2rcHw(%{Z$~P-KVnT<fPpN13e@)uFA}eyvszE2<nP@e{uJuO%Qtj)b(0E)eFX;tpr}vRm3??MvEfcZ?FcMJZDD#{^thI+8gW9A@{a<+eQ(c?_zv`89jhoGs5vKDf64ArNFr;(3a+4ls#)?sb{Q63b@ZmVPsY@5Wsv<r97-Y%`rR{gN$(*a-Yt<4@LdNc~baAy&9kdz@XT;$rNm|7)v)mMbp_)Q}49W^QSXHP9J1rb$eO#jwn=NNxJU=gHKt;&#JMGxgbmg&s9RS}nH1;bB{l@Pq|x9Tr|y&>G_x$@XKyee>pLzlL7BC{sD^dJecDU@z@k`3u-GP6zd8vY+NWcc`a>zDPm^s&Sp9)CH%rmTj+*@LLT>#;X2;Rm=8!Ybsx-OkG1|$+t53nn4#;miW-H(!VxjpHT}<gO`=`rLCHfpVOeN+L7PuwrWeQyX~QwMps5Y*9fUTf4R_KqIPvivLh}%_(JM)c-}MyI1YwD_*aLEXq)>;rrWj9n2DO+7}iE)JJ!<j<&ngvJ}GZq*K7nKt9$f(m%cdfpn~c}38abiPx6efu~QP|&cN0xi;#R74f_M;_bN(Z*c_vPs2KAsCkCf%Wl07iR;{v@O$|z#xm#`eZmjdQh?0}5beeKIG^t9-($a63Spr4UfR&qTx7(2LBlmnx1){X3vD&O`%KqVMoocUL{D;j)wZz&q$}vLBvSm>?6xHr~mDa;V85?2pYSXT1LsYB=t2_fLOrRpY#>6QOeKoQpr;NkeT>YZf^gSYAdi4mHiG{oBGTb_Rap=?oHFRA5rqDlYQhUv*8YIn|z}A^g@rd;L74j;PUX6XWFum2sOgb?6%Us$pUhBJAjik~u=y$Jc+%v@NW@yvC=db%5htRgt*eHvf7++FcC&ooe8<WXwbQ#(gF-byPl_S>(p|Htc8H{<Lyr3Xf)N$Pt;(M$$qACCF@r&2Bm>MI^Z!K2bb>c{eI6wc0nzuH<N6C4W4NeTw9%s0+N|m{DL73+_{=y~9ZL2MS%wh(l-G=Bl#v6~VTBzHSy4$PdUhbw%&HNz_r>0r0Tux?-uEU88w$-dDv*0Yeh-V#%Ph#V(rTKHf+#D(5D*`4>R`Ub>Rx?7^M#2pkfQ@ZwyH3eoaGMjVj)wd0770F$nISUK6U_z2t=kS4b2q1c#O4J2wa9&S`|nAYmePd8nqMz^nH5-y(+vxe8!H0ok_mIM<Y$`!TU~583k7kE6*ZN;E6FiFDUZ5=@~4X2fnn-r^}8WYExqihLTxvmTu5WYv!xEP2tx~q+tL(}4XpAkqEkqo$f^RTOOywi6=7=tX*~6O^9BZAuDcd3kvHJ76cgT%P|^Oh*sl`@P#=RRo2}+)r7#iFN4w|VQ*mX{51bFek%K)bYuD8A0H%AcuBbtYgS*pcwbXopt0Kvq+TE=3I<ActpK*&i-S(a_z{T?1QQmN9p~wRIN9{*2%A5GP<h4^NqNYO$W?9R`+WVNCCs1!>YDkJc^qjhj`Fzwcx94}Oi%M4%2ZNgaQdLyKQUdjamu+cZ6OVbDIqoXfN?NI}B@+!sbPk8_{Pwji&+7wH*&Q0xy$0@4HD23+vEpr)Ox|j5PiJ&fnT4w7lFzL=g<icG@EesGrFy(8fXZYr)tkL@*90*7<Mhy>wwq!Z>MUcm@ZsU+z+$uO>Qq}pX?5HU5yZ$>$C5~O+urWnvfNrp8ZFr!OvqOoC^ihetHKs5ae!HtEdm%3a(`RE*Kay?P^M*~ZHpjaOj)~(z!%E6WU#%E__?b^J+cXvw~Zm_gzG8N+_*j*3uh6><JF>M2F9fT#K;vdk`7e*AT2U5`+mHGwA*S&fCUIwtnrGI<Kxrd)7ghtCyrMwDY}bn;VU-=3ANEQx1I=O)*CFBh%H~hGFYP}@#AQciFuQgj2MsurFRFofpnp)UAwD|wsAPoS%+$-F&9&t?!lcFQo!ep2Tt{DTAzJmdySv*HAA?A%BwD4;@?3~ZT{o8qRC~(3Wna>4A{!3=Ai_)t@zgcRqBToYjEf0sdt}`SZc~Ds!V&^Y9eyDW#Zt(o(&!2^`#;N)NVarFVq<S@YMxiZIRwXjlSE8X}Ix<=vrTrRJ+SfEd$AVpxjc;`vi9>dmH9ru)(w-?m<i=4z`bf;2^-U`8UNnRPQ#Kr}%7QjT=jDutJ-7fLhHgn44F#f?189>Iogmqgu(0!5}UfWO;Xvn@W;p?teUfb^Px1>=Up@T3C=x8<IZr0ysiFeCz(2C8<%dg#`JuPQM$|f8AekJrkzL3gO|6;EMma-gaJG*&<GTGysq|+d|q`oj3r$J6_Efum?NHq7EUM7Ds)gv7Uq^a*p!CxuaTPh-$>iLN;&F(4BnZE+8$emFr~!wL#M+)hnwlmKb<zv|OzF2RpcJ4bASX2j^bPds2-h_gV6(37#6_%m+tyLxoD?qs6XtR)k7>8=969slPKRM(yLoGIczgjOn*4ecE5`SwUjwReq&tjJxMZ5Z;GXPH$vvF9Sb5{o~!)tG5!sBk8xf%pa29n`1i1-lrF5G%KW1V{beym`3tS<nhD25_<Bx5_$4laX{Rm2?xuU6)`ftI3`BKX*@P0A50B+|L0wNqv^f%?VEX#<V^|@t%b|Vt6=MbkSDa<;&h!XiK3WEIAJqC{r&yfyPrP1KE^EdTYQgUtJktAoHz2=?lQ*vfbkgJj@OD;0Y<AWq|Dv}bYM)(SWd5zz=5wZOe5mwYaL04Vz|pWiHZ%o((!tv%AMnErYx+W6se#REmZHWKV3`gbzSw@SfwTuOcmXJS>IT^P2$;@;5pJBTGk`G44a>POC#UQ$z*cBY2k2(xb0kn`HfJF3p3|PBk$#;6c`tf(1bIYf;6?J5U#^yhA1vRtyQ~tP%QAzxF|r`-%POrXO)(<CDsSQTI}`XjO0V+o-#fSPa@(yK3-v5U@MPB<!Tu+Gg~H?Y!u5qC|l0N2?RM02Poz(GE&)uHQE8w!ZGO6j~{+IJ!_pCsnuu6oRwyrZP3MjKK=*H%)$OayF{b%Cbq<mvPv3S1Wy(R)Yvxzb28%ix+T+UJqvhrSc1C)xzBso(3mb(V=Amdr}}aPgS*_RS~CPUy-wz<xhmh2@&cpdb}30!m-KbT=CyT~DJyz>cFDPdyn5HFCq*J^KyN247Oz2nG2x>yzuAGgf5x*vqvfwbv{(gLoLwRbXQln7Sk;jC_LtCzTV!+=oFo~<4q$iyZBxR4XBpo1tf*m3*4f*S$1h%gYSq1eEyCg|4z5FzbSTYtMYS*v?6=ZC5UJ%Vgd?#0@pW5&sr5NYWuijnu3-`KmN|%tzWCTaXld%|^6(%+ulPlvT}iy2>Rn!kaOISA$p>E$x`c~0IW<z6rSM1vsx|xr?5N<`-`o6Bd~^Uj?`oD#tH1RGJirJXe^^v@o~@RV+VEqZg2Vr`2-OsR`tbLUua1LPZ;xM{oPKzR0OPkWTIWU_5)Y&KXv}jRvK8wK`IL`UUDT(~l=4sG8a+7}#`_l}qJ^>s`1dQ$j?I$jHrQGu%yak?GQK(s0e5_wjF)^+=I^c?l5+Pi-y_%HU#b|Z>~tS_r~Jj?k{#JnJYt7^1=2VcQ2Mwkki^Z&w!0;mZ{?eucS<In4S|X7PZl$*ZnA$&Er+s6sc4Pno;pgZJJ|A*Y7*>dL%|0x9-qJSRa~1Q3oRVlyCxdE4vtAQK*zYjArUp;tyh(+Lfr)^$O@2cg+*oD_{K6op*8CpZ&`DRqxo!<QUe+kl?Db>h^bPt9~|VPFN+BFi%#LKvDTJUdkEy`!xoFLUmw-<38nVhz2AC<+gNLegtQl1^B8$0*Ysdx6INtev!1Jt-I4k_!$r%;7qcmsvXviQe<}(rqr;h)`LuDTBAiAVQa4B0)GlJb@%XH|!(J(9Bve`ZVJ#f_GPFR~Mn)xkA2K2lPa9U*bIs&N#J<iq2CF8>WaIs)H4@#S`y<;q94{Rx56D|2QaJ%;n?;nHu>9_$o2u(iJm&~!ew9qh!*=27hHMHv3F7&Md%P@_J6!ATrYakQv7BnylKKzBU3Hw<<)LoLtw)COS&;>aQ`2y3gSrOX71hTbxA!Q}L<#762?+p<Gr_3TaB#1hp&%uTFlUvPPY*X1z594`3DiC2Y`0QJ!NU)|W60%jC0JY~Q2BjFrW=gV=0%`C>bU=g^{Knqj4P9S%c=Goga>ZKYuQ$v0`S><$!Ap=!S^Kx6<tXk+2_H#`qL=qh-rLpZ{|X}U(~S;Tt*VM9C>A114Q=lf@bzzb)kIRG!>giShb*N$bjC+;;~*zk=G3KTO=YrQ8u5$(`nzvs6TwK`xTav^m#*)hK=Ix>JJrm3LBdRUY-3sRQlngSu}h`&o{_J)s_3pu7?1=L)cbRwu#2e5*s#kjP+6$b(c(?e1DIC6f?&5LL9md1%0$XhF@TheF}YdQ$bybg-+<?(@K+*Bn}Hc93uG-<OL9xP7q@eaavk3&disDQ$7VTI(P4@Rd8+j%?3n(K2s&QGW|NO;=&?pP&hy~zHO<E{HDaN8fbU5z=f{9)KK?!;-X>D^lBL&)vic-tes>hPd|GZ#^{Sgc`jCzq=ZOuE)2KY)4=#Z*}P#-ZEK>5O$ru1r%G+8<y!(q7ZpZja{o0tw~_loluc??5exGep^b)U%*JWP=Xd14IIT0UlIh=-7GfKs6|Vk;@?eqWi5!kC(`5=48%Bg(R=zWm%I*#tHld4zR0QqfQ#sulWlT1iK%+AwWI<15X9%TLYa7TSx>}_i?Wik_1nN)$e`0`W&g;nMDf4njwasN?2DS}!i&_)U;z(qzki&>OeXj2^-CWV`nO`RG(-0K}p&!9Dr!y>n@>Q!)G0G>n<+a_NWtV}G^LV2gP4wN7^-`Vt;%;vls6G*^yS(*Uc>UR|MhLDF$tOocQEPdKCl&APpt~2zH~_&mY<QcW?doTRa`eO)$cT`!`QrL|b;u#-s>cVSAW%3Ec-s-M8Ft0|j-(Bdd-}m*vvpL8PAQv`aPWP;M^@stTEb8H2}f#C*RfM5B@&LDP;KI-^`;B{Oc!S($6Lnpko36plJ$q;svE`0Oz~|Tgz6JF^B0mb`KbceaPrSm9Tab~M1Y(rpM~>_ap)yOK8sn6o^VfA*Rbr&#w|d@hejA8NuKAH2d(B@XHO`Ck*Vi!tyTC?598|bx9$G7Wb8rNdF8lMtQ}znq6O-`BsiaI`LgftaZm?gOHIAv972g3SjQjl&BEu~6m2XHBj#)tE^;OwR$$|uA3EE_s*z(;a<<%6OK-bZ+h7@%J%_C*PEhS(N`6t_u$S^_vIz1;93@0mGKEJ@v=(o2^A<xn#kRQwB9Kw?E|&Vv&fIBGFHS2>7pdd6-0vzfrem&u&9RBKR~x}|`3BgfYJ)|gzw3lT1l4ka;{c=iyBQ$oU9<24Rkl;TTCUOv{Zenu&0(@Ls~Qc^%d$<)Sb#bnZ|S<_+`Z>5_%;`C`AC7Sc8Y^|GKnJ<QmyuO8(@CRM&qWf!v^qoY(a9wzE-?>8r4^WS&1PVz^}lEj@hhjoMF>Zq|4jhBMH9>9Cj;Qcr(O{Fi&!Yq&4+uMoL*q|3)$KT4MbKaJI?=j@_as7(hTgj2;m`yo`wl<P~n~c&N6=8c7Q=`NGU~Qtcd*j%2z*?xGun%S*)icglU&z=%@ax&hnJW<4y&%Jfx6-3(P$W;Y+!mXMlFM2JH?trvfN_x}A)JAZxI@9p$Y-$(`_maEi{h{fn%kI&wEuTKBodcfvAZ{H&o%s)eQ?7`V<w%$HCQ5ZI*V@@}#R;63bcq!*HzGVlyx!L8KXd=i{V$xTc0m$uwG^t*zB~0R7PJcJa`>jz6Di}<@3`Oj0slY*FIz5`A7`??PevK!FEL6wY<p-iGsyFpu0J8`G*TsD>Fe~JcWV4Dr#49UGQm?Oi1e{gu?D*4}Iy<Ab;Hgpdjrww`539%*oPyyC68KaKFHPfTM^<{Uh)Ft;)yN#!dhqqp{{~P?0|XQR000O8002f^r7%3vmlps4TTK7}6#xJLa&UGrHeYRZY;<XDZ)9a(b}=q+dF4E5Z`(+g-~B7NHNb?7DaBSF=%$CkrgL;M-8njO_YR;Hh_YCc*rrH^Mf&K#|9#(4yh%9;W`PBkfGx3(SFc{ZyQ(CFc>nH?)9<aP?|!%TWn4u2G*niUY*en2wX&|GVrQ*)YJHidQBuqjSwvUL`tao3l1XU2fzNfC6uDd%dVX|tzEf736;YbV*aA>xjKFju<5<I&BDD%=p>85wM9J2Ad4B%R(kj1F`P_O{Sfy5)T4;Ht!Xq80g{AfvDh#2urV(Q$^F)Q#)915XZF5yvSsF*{Th~gUajr5Y3#fkb-iqagievaLHR3xzLS!Qiu=i?~r8+`1t-aKj&?Z+f4PE5rno4Qfh)6T_2ijW8Qsca!%lzmApcJc_Rv$~ERGyYJCm5kfFI6&&)3uB}>q5p7>O*Utr&>1ySM9IUG7gW{X_*17Hf2%f%F6O|pW)27Ezf)}FBN<NGOlyAR|yc+`sEKm3UI_iXa{74^7)Yv;%EaftRP^BL10DuEX@l50CbWB^{6V+x5+w6tJgnunpEHSve;GMb-KP(MfC=pqU!IrnW~|cWfbz@8FY@Li)!R|Em%dGue&evB$;o@WNml{*6amC*+^iYEUKWqM|$g`Q#a3*&R{~oJf_t@KfHbO-W0mlhay*U55%sDj*i|x|Krt%S8v}~z9sVP>hycV*|S;*W>*U=yO0IYH=Hd_7T=%zxcE*SJ$>@#?VDFmpZp4aZ6R~vT0+Yr*hDuftX{5U9?6SX;X6PP&e73Ps5U4QD!5d)_C*@rI^K~54~d*_0ryeqe5(puV4-lVG`DaFXC9~5Dz_b&Y8K12vc(KR2LKpBL>mLer{M(y#s?&%%#$8c*#06-<JNdrG?5oF*2-}#1t0BMbq0wNn74sf&sRA2jM)K!vI<~oQnvca2!{Sc#-)0m=V@+>-xFPCxT+vStgH}(X9??=zA**Nm{?#)EejL|V5a~D){a%OEp{%5b`qx7uC<ABT`c@JX`&43Aa1Y>%#AXa$yS+8Jc0rl^s&)=9YZI(raVJJ!N+MKz-Bs74KQqO;#mLTTMV*kF+OQju$jyY))m~x3Zo6^Wo;4flajG2^<(&erXi!enTHO53K~E%3a%BRwEA3&Jm9o(C8j_C%h~CQxQ|M>$b&NL?*xB&>aCo)Om1z*xo;|OLA%E?%TU<?K-5%w&~*h{5ro!49*KPHQz@`~vNcq`R4QZQFOj~k<*lrL_}F^sxkU0+wHmy>)??o~X)vSyfA0Z!%<mNdYlare=KdX~!rrEv>5Tmin`srRbrFE{MxnB^T!qmZST%^#ZLo&6vD~}XUgcXAtW#b1C-akv-=Hl{s@G*6sseO6$oeKuWzp~%Hk>ErUd1+X(gMkeEJtg^D;nudltcw10xJ~9ofR6c&a31~#!+KeaZM`2%7_62ngv=bgDxuEoy?TIa6lO3KC?kTy4K>vnX8-3V7!&MWP5pI!<0eZXfkWHJVi>w=gG?HP}yFPFi|sr=|C$4F*8LC2_5UPb;2s@)N!$qSqd`Nfh=!q_%pL6JW;Mc3B8LpMF9SU@L(`^@!j{(TkvFqG7G_R&{*ZT9gem@k<sD?+;o02zgE$9R}h%*zOMlF8(l<FS558%xoCqN2e@bey3Sq26QoErxmchDaNZ!{uxKQ^8WcNH0+|KBGc69X!_(T|vBFfbMneZrg0?1qsFK|_zpw{pm}CP3gMo9I@~Gu`*g0_JUfWJbHx}siAdXAuvYJy%MwdDt;P0x>kjD@6IrNL-NanW|>qRmn0H`w1##^<MicUXCjclLjCA-StDGOFM&s^)bm#*~-{Jnf;nmiLLm%jWKynO!T*@x=&mp{7i_!2yO_1b{S_u2q}Z$Xmgds6{=(^SAWQx>bOT;B#eC1WrOT+1)efCV*{T3fFxR|ZhFaT+Zm;Z%bFNf`t-sIZL-4ryArY*wPbR34!F)*~wzVsb+2o}+y_T{3wK;v3cm316t2^$0k8CsCVZ1-KD=nFZ)FqrUOQIxDdZ3jOoE1U1U?p@TGuZ+&)CI^#&Bgc*!I8dyc3N~lG*Ax?G<Tma2cnGfs8MN`N%=p4RDYgPw<GlB@N8t4jS?It`BLqO@ppI{llsRXT6CIu;>yHQ-AL9&Uq!v-KCFtJh3qIz0NZ)YZ6-U!DTG;B@$)1`OSm>J{F%u&vxZJCxJr7Xto6;4v_mIQNEiSXSd>}Dd^iC+xYehzuCQ@?nmyV`8fU+#{%6j01Pr$%Hw6@YQM2qi#Z!vfuw0?Nc{f@Y>$V}QC{_vKhmqnmAvpc<_;M8l!jZUjBFaZoh?Ts_l(+m^Bqyfq2*keneu;WMp18-^#%uQR2>M;}Y5D{i}XU_i#MTy^CoRB<J|z48qO$}qt~Zb1#~qA*m+2-?PpcrlNo1mxA`hP9(IJ9VGc4wY*Wv^hb}!Gh#lbW>)hUCjSxJpr&3ZlVZ+kbpFVDg&<*0*l)928FJ`#kHiRqnmC%07|#;naw+JQM3pFGLA?Zdeg&6F`5B^LJf*>_WZRfc4>GH-)*Wevl-EHImi!$>EJ<JfsX)jgl|=G?@`AR$SWDe7^MZcNSe5*M`Nl0080!<SA8W_>z0G6s%qF5aaLOxx4I|f+>LW3p)dzK2n%4hg3}?tnZeMVuP|Ho9lh0ok;X@>L)lSjw+~D$X{L5irh6)_?i@fY3r;wN_nd?hUk3*x;UEeb)M7&VyEEtPMZ3NgYU_rN{?j;URqBvf$#%MY7RLb#rtvwVP>)68)kBU2;M>IICYYsu?MN15#V(dNm5GRwAj{J~?HBVpm1x(1U2a1)wSMRVGQl@R?l`dXmBNf5z`L_=D)p0`4jJhqC4#W_qrG!#^<|P5;BLMIH3m-D9(CfWd*UvfWfT6kpUWE3<p`u-#vlz~T-7kHz8c0Qs{6}1XJ)}~1f>WL)s?b;d->=QfC!_#e=;?RE9`vrm7S5<(Ww~E1M$4N*`Oiz=q9|F>$Qv_J`d9ZSt%^nB#{i&{^(xj?p5xhjM5N3{&T}T%|JZ|8Abe~fF;DhJ@WNw{A}108T+PURR~^qBB?DWn$}H%%_gy#0}TT%oKeYUO;#2xH>L-cVPFDXzDVY?Tx}u<sA_@lhiYa_6IxWwNzlG%AYG`;j8ciW%h~aJhNO;l7t`a*#>TL~j6Le1v<Yi8U{Iu(@X3s)@}-lN7>9R;#7_-fA@s(yu8XSaA_|!$Ib!NXbL%uGn|=_jb`Q+YG(boEl^P;_sUnLr^S)fHYS&rJV1DC<h0&>INIl(ZYLqtrLIxzS^w8?aRy5J6_=}`}tK!n`JC^+<ohz49X`<Cw9VK&CJ^ADD%s_t)yDbiYTvQ+@NZLwL{^bti4R<1f=sJqv=z>#YXvth=JY<qKg$f!B>Ckn=Okij0qt93EZ2?}3o=-aF^AEz5W`DpclNtgU=>k}&6~6TM530v(z4f<axY%sS^w>7dqivMP*f=UNuqr!)bzWo+yqQvV7@;_GRBi}qdD`=&2|yd(a)#NR*H8WxynFxl^}BPi^15B(XdfZE`zW!irf;7A7jOt>5jz`0=*e-~w9~N1FSMGJm`3-xwIL;Mf<#>lcTg12JhgDY0egbe`le?{FUk<wn`H2*i>MOXZarF04xmM}FZWgYn0hb}w}%uscZAdl4SeGC07Y9cSrB#7&Q5Q~tEm&N+4t)71xBGrpZ5oY(`*x<rTE0bEQmjWd4OdPVEnsaot5aK{DAqa4y4;i#i8{{kn{)-1Xtz|5QrTJnDBUC6@lS<BKM%cT!DMv!aH$AvCAoo<>8{8zkF}9h-03+0BV*hxRw!!I&_{xcbtXr@V_e>Q59bE{q7!Iaql#^P{5LaGQ>_&c%OKC(&M15VzN2SOXJ=-2Ta0jB$AYkV~ksf9@hJQZ{HLGog_^n@Vo*Z4F3aN$SmYvIJPjMu<)t?cpM#EOjwc53UeVYNm}M$Vj3M1fU5>l#!K_z?eFiOJ`bM0eE#&8cW+<4IS)R(d?HqNy^J5CiO~!ogRb9(<3#jB&aACURF20N{N?Hl9jB`(EwwMi8EUX=nTOgxIjcOXK%-~bgO2)L`=7)q@y-JN-<=r?7);elt(6AnHc<$jJYg1T7-5hiMJvFS#3AfAp~sbtC}>zfIt5YB#qB^Jw66(JMW0bM!bm<dRH#$K%62!(mMiQUn5!?NEYO)+N5D+1tj1Ev_+*iyT`mKBi7{V=yyLBq-Aq-R<LVa7bL03do5`%c2&IY-%)n#GhYndW2}22(?oy7h+{_s4schOn4FIf`5-8LS)Dg*C5eW~8Ra4!Wor3a8L7d><-5{-9TP0Kg!&uq4;XZpMt(f4fl7wUb^sJGDvB}W_4M3wW5TTdBJN#@`CNd#a!fK71Yw<kN0dAu?-USfja<m7F1I(B)H%{yh0zDV7KW=z(=mfO}<-PsN69NlGt_9|DqsM+_9PyPLrrM4jz>C%TP;gOh&x~SNf?ikwStrwJ$EW`=ro0+^lsY>z_d2YqVygDqbp}!Kv>G3-4CY@$mX<-?^)8B))n&v!($xR(6h=va`y;^derF*^TQWhBL1mN&yP+0sQ5A7Dke$c=g79L5XWeN)<#e`^MBQH!QC1JzFwdxJIa{n~Cv6gfL6v>)CQoO;u;QLx!QP}=7M*jno*~I%vKMO3>G|QqAl$7q!es(GMw||;oOLbT5`B37?i^xlgWDncGBx~qu*K@Q$t*Vq3sxxT4MG!bPDl2O?kB<I+I-%sR5Lh)7$85XlTcP_ed~RV=EY~7?!oa#7w2TO9<}ohH6NopO^pdf``KF7%xYTG_J!|&+C9j%tgTS=&ZgC~!Ddt!eKvrDI=&|S0XeAKcx?fyONTNv&amuI`c1I+LoaH>od|9u+;1sxJM9N<`0&wOAF0r>)y`R37Fh}EZn1jEWoNX}5n>9+UxHU$m&4q(zg(_}v4VlwA0~~-UkwmfAJ5T@Br6QDy3^U_N`<$=#bt852FRSZn{2M<vkZmWXUJA%uEul2h-=p1%B1?YxhT~RCE2kh9l9ca$ee^G50(kMj8+rw(vWWWAf9=%F}n1s9g~%hgN^nvYN@Z<Dj4_cwtt&y`kVC=p7pixgbAH>+GO7>AW*}j+jyQP+C(Zp<D-O(UIETKXJVsdI~q^|P~5vP$2cLRkud<&uGR1>67~|$MxtUt)Dt?~ZI#J-ha?Z@qk!wP;l8G$u{JN?xU$3abU-JW{cmW&VGU)0#t{>F0_RxeK+L(iN#&ewHo#yFL!7u2B30NpZZp;d$KtczmC2=F_gXSkk3WtZ5aWD?59fu=q$d3<d=5_c9*=|4rl5nEE0@cPp^u|RFla8TRp+$kXJOssl&f(#35)mz*UrP{h4!D0n|I9g;*s!epwmq;c6OotAf|bG?Gm;M50IiwvkC8e2!Pt9g>oHP9_)nAgn4E%HdN%O$Y>i6UK`Enwzn@xtsXYB_qvbx@MQe<oR&bERp&_e{Q$u5_|w}MpFxk_?#3h$4g;>Wv`^s4*zsF-uwSEmuyq11q-}%B$#0jK=LScRScXcFWdIy9f@u+j0@OL@G2Bt}g-<J~G^Mz6wU^*iC0n14GObk<8}steKAqXw@u~C3n7;8bTK<92fgKy=UaDK)9HxpqyEADwa-5}CV=&>5_j92@9I)eBw^AVeBqbm95Qw+%T14O?RTqw1i_dukPdliQ=e&zi|E^H%tW`;Q@8oWHT+i6d&kub`*}1i|+$a%d!2??4e<hoN&jD{2g`m5lDwdfBULl`U&3<$zU4w4FYj8;pcdE9$x?2Z)N*YhH{+n`>c;XZ@x7Dh4qZ3$lo*<_E%~wpmT?T`ukE-g?{CBl0P^84Q*d0mW4l2fBQ`=F%wNoY<{|qhh9zg}2M5XBFEvMk&PFm0=2U--juE@cELZrB^y`|<nu3L|SuG@}`(ShUA23W6Ef4lA=om*mBm$<!cwV*lH>X|*TQIKD;BQw6m@}>JUSQ-bFn+~gIzBQ`IY*gbOew7HCTGn>{#&qcBbli6I*wL_u0A5HPX{OE<UbNv8-NCxrJrA<D)B&Alg%U(R0L27F%}%)J<+Ym2bVN})y*9DfeA5%N=yYc^sDc~A;%x5u5UQxRs_Ijl_w88`{Q>xspI`m@>+9LiKb@Y;PT#%YyK-o%0L~gs)6dV(U*bhy@ui!J5)k@}4(1Q`#yXX_M|NNXZG6W~(U}t#f_}eN-Pi8FO^upg!CC}50cE&L8%TW&ct}&r60@dT#inV`dfvB}3QAnK=F_8(!#$9(p+kBpJW3r@TP~z?jvF%rh}GYP*D9ab<!r|Pzq1QXu)5c5{Pq<Vs;5IJNgm;yTyq1a?X!e_=W>kgoGBkcssz-w5-TX6(<Htco-b*DR7snzdelDscl+|`*W!6m>RW&e@_b4Ez#|>wddUZbo4F%4{W6!1%g`|iwEO5*;<kJD#D#}<%h(DTh4{nWU27ss;MYuEL~(>!2cT!`dI`F8)pg#Hd$@#XMIygy$Y*}rZ3bg3vCRryl^pz26qFi_D=s8ZB?-;B<TA%<X6)-Uzf^gETz1>;G9aI*>pGAP<wnY-Mwt+#znddLq_kE0QUadr`GDrZst<8LV)J3aN^`DeA8FL#VgxM>{x~qkTH`uZJYs@^&$h{p_W!zb4x`?(s7=+EoU|AR#mV5n-1LB%qF*44M+n;b@+zRsKGbr}=8xdZ^}z*7MZhQ6gW@_3NWV<QLR~aHF|^v@EhwsJ9pLzBju3^*qGFE82PRn3<pxjzDapY@1#hLvvxz%q*jAJTEX_7`HFg^bgtcEd+OvP%z{ec8n!1XB@pb<K1utx~PhvLZ!=qXgMg@A_DIZjGho|rQ%CD|40ie-Ua{aN4atNI94FmGA+*sc<k<b9S`O|CtjtGmgh1$><yAp)r+r^J}XFP=lzsi2Dacx04gA}7a?4G@A84TJiz+ae{<G*ih2Ne8Jclr#b9;V1|$L2!U=+)euX(SSIM|Tj^shhRR3hOz2U@BK?Ec+q~ZNw+MuwX+!yg`X3HVE!K>k|~*iI!r<03+^UJT^<@ku$S|xdIAm%wW~bK^L~rjJ1AhqP6y|Tf;PUx;5;L9H}%np!OyFyHJ*-AJ;HCvX%M5Y|F$EPZ$vjtJ9rbmK-_fImF@h*nv*GET+Qlk<Mef$y-wrT5V`i=qD@c6O4m{i_JO0c68Ab06J!4fZ_bSYEx@T#VbR4`ZX1nPCS)tn+5X>9oy2$TsF2%(P3TMjFRY2RmU^u_!})CF5M;WbF1Bu(8cRw^v$VQ*JzXWFQwR@$kQb0uh|x55n1-mWl0V&f#H99(n|c%Exj`7Cy!#Oi(zV>Qb~o{BN^}0UoFyOm!%P{6__xCotXvJeoim)r#(V!(8YklO4Ksbn(j^7O7OG<H&+_mRkKp&M|I{!8nEwd1BE&w!OdC4@leiSUX!cy@F92au1XuE(Yx+<uu@LnhH%3^!}1}o|1TVidoP1y<)6Wwtmbh(^RV1!+Q#-SsO<C_lzL)}=dt+jMD)o%_#q<OAmQ#+7|DPxwBb#E@UAGNesnWSLL5razM9H07I=yd*6b<%>{g0_f0EAdas>KZ|C&8ngFT)7RN^niaMB@3#a*SFvdmn1<qy5|R{k>Q9U{8w-EYv|J<BnPUF?KQxsFPm+v1V=@c~ynB9(_<n3aT6|BAs}hG8JHh%d9+4@{>AZZDbxO5pWY?Jpnv{QUfZ$>WXzps^mf^wH||?(K(z-59e6E`LsR1RM3hAP5d$7sI;2>)(GpfA#w9v*$q-0Yc;Yg6;nRP)h>@6aWAK2mk;8MqQgMI}YXw001v4000mG004ApWn^DtV`X1rX<>6NaCz-o?Qh#Q8vpLUg6H85Qlq$W+7&}E9l9jmFkJeAuIDd7K+q&@>mo~@NI7W__rKr6SCO)uq_^I1z|Fu`>EZLvhYx>b^bI;Mh&U#9q~YkImFS|Wg+Q}PHud%s5?f9SQWl>%L(95BKm7Fb)zusH;r)F23dsVfYfji?^bMNb6Dm=|NqM{@WeX!^y(*+QL!Yk~+3p7Ah}ANgT`z7jbaB1ANzswE4UbCx1K-p;E|6Svv~6nEQd!g$BFkk_@#8IFtf*E<HU+&!*F2kMC->8DZ(d*1Y+)WJqh(WX5XQ?^whhM^6`O6{NJOftmPBb2qmiY_x}j_T(iF0;;<rgjs#Qx?91-xT92w=bm6)moVQvyMp_JnVp|_Af<_zM^#?ubT6&2(6oMFg`6LX}BuM`<WzLf@i%bSXqhZM;Yx3VaO(*p01ct{t&-;PG3*EXDg6mrD)65&#S!;p5XHW?xK&0tD0JsTOi`UT#uYI-X&J$8NuIwuseCqZSCp?RNh*&}E%0frgJe9`B21MC4Qx;r36ACl6c4UJdr29g314Go@Oy}S4u-iy+Z_hL2V!H9qQ@b2&L_Z7|Nv$y*K>6w<<77P{!3_;1S8nVF!OUQj8@{`ni4m@~axF{=j$+`35Jm%R2JaZpF%u`f&)zs}4SgpJ0S*Jj=>T#_7Qr<m(z9+2NmPI3^_st7Eb9w_^0@{b>+7wmNMRXkM$edj2@}ea|D@Q0vlt+Kwb0`H`XsXkpa&k}zLJOrvTB=?yg*KfStIu4r!9HY{G|(4#a?;I#GJf8PI*ah&MCZezb1gdWL~SYb`7TX39B*YKfMo{ijF;DE8G2h+{02<<_2x#|wHnIMSvbnHD(X~SIOg2|Gu?Hn_6<>H@<k|ZH^%YL96D2y*u->UW{Am8Tb={UvaE@GacUw`8wxbd3i*It`CfLh+l8Wj#nLS%Ptu7o)<H8%Op?-QoCf12@^m8Wgio|r9WRIe5o>O^itG|9mHElMe8hH!HbmU!{VDifDa%j|+gp>LOimtiF;?T#zo%v}l3-5Mo(mq(%n*ZUO${97F0Q>Q2`XiX%N4*5H{{MVi>K7Ql~K`wkP-ggt^&0$1({I$kuOl25F^>oK)<pK^f0RKfRQIg&d-NU2WVF|1M;!flZ=aMIZjjG5_Cs+VnZ-8E<z*xy=@V4T?o9nH=b?CePTQXcU04#+Ldm{VW+U;P=Qv`mqOa<gSu4$R`rp|-%2G#w3~$Ad+jzk!nYFG>w!PqJ|R5q@hDKQc232t|2=Yz5xVF8j%0TeEn>~xt&5WDOv!pvP)vzPI=vZ!%uQ32=V;n3zAX{Lq_Jc{K{(Id;^@~xhWM_~xDjvbvY?;xu_oY>2sySpA-~faJCu}wk<4e(<`r@W7?B4iL9l1{absQ7_jJ9A!-nD-_0gE}%;2>(Z<|8Mf{NsyTGiOJzVR6<$!5U_BFzf$jQaoe_;P#$TZwtL3l%7k)01rUuQchKNr!;p;_eB!kiBbVhCYHNE-gM}iaaxb5wQ#PpL@}Q!yvK`i1tvWnJld8Qfb2{eSu(XdO(&wAWI*T{XN0!SHqug-r(~el&V3N1Mt*L^AJS!`i23ll6yOwUE+(roTi@t{KGH50rM^}UANV(cm($D_SuGz@9r?}bs8RkzPqmne0^BI4g2oC9Qp44z`cLq-v3|Rn?bs_Ev%-7(?Y<WosSF7`S^bd592BO&ml1gKN%GdVqb?118W8znl)#!Fbkn|1W-Y=4^9nC2ozf25GW0G9tC*a%61iy=lvB6V124EpROvCtRVGI1?oFe>V}_a;jrS)Q}mZ^X5u`tffI&Ol(`Y>mfj2MRt7mx5XvXOPT|7N`T1j0=<GhrwOa44i60t8xB2>-W%lw&9R^r*-UwD>hg1O+C#+XzY|*bZR}LGZVzX+M$1}4iC%ohvUP&xJ`N^>-qWwe<oC(A6`N)<py~WBSE-~*ms)Of{06W5t6A`9bVR%oQbm@YSvhMAXU8mzT@r9D^z!W*&F@*kf)L?5G*nD;!<F04VH>*7o4M>;WV-GLx5138I(9^iWF1qpxqoW-N2&ScjWM&p<e_CiXEr+MZ7OI4mED|+OX%>yKCns+xfkwm7q|ulu^f(@Qa<QmizhDl`SU~1I@@4#ba6-eua8lJzA4B7_-+$v2s)c(F4r}QopP|#9;N#Yl0VQd05{@HP!AE};N%)!h+tB1AuS5-}8QLc(wwRWp<G)4(vp9$lypf1cDpqWi3)DZIB2qD}$&;qBj98N`zn<P0Isc}_-0O)RG)meH9NE)YMIkU8zGSg#>sBPGl^vpL9lZf_Hid&aRD*R#ljbN7JP1Go#L4gQvMo!1!dcq8O7mD)r>O@^%>9y4V8OI4K-bo%r{ASnPf1%;-M$EuxPmp3Y`7!=bZ)q8n+o;SDK1we=i?jIhC+^c%rOU!%-HGQq+BbbH(iG^kaM9gOn@~r!F8(*Y=iP(`9W&++D=X&I{z{a8XFBc=nw}O3V5Q%EGFuV7m;n$`RTW>d|%EXzy9cvzc}?L()_z%zk*{V^y94nqJ9yQthmTuxE&%$IJo4NmTTi?eVN$;=+awO=BPJB>@PqMi-+jyC^JB|XmV(%;+$cyv8d}Zpcn&&XzPg?9H;)7gezsTEBIY<JpGa-!!*p+&gg>2WpT?Bdz4{xXpp+<cgG&J85{5Q5e`u!yciNT6*;s7y%v4a(2CTP$prJ4zv1$LQ|haiZr$OhEMKQ6jUGdH!nI7jPI^e^{7ZHIUi~57Umk4_s+clYo&k1LL)={zGd1<pzb&G6a;Z8>Ac*5--=l<W!>J5-=^@^)(WYGE33|=b8jnGoxvBNwelhoTJJh6JMosGFK$B8gP8e0E7J6V!DF~M;u7%>#`(%%?TQ-^(x*jcI3)YwBS4G81qra%JAJinKGD$1d)hC<$*jD;a2(~0&3-zH2h2mP*%oncx5G0}UK8s4o)=hj>WZMI()R!fyw7ca#g^q7t7xq2vUHp&&0{IByv!<z=WX#c~6%s8tT5hMWo-dZuSHYbLSfgKJ&Aq$c^N4Y}Z~Xak`{FeAfOl}{Pq|BpLqtOs=IZk#Yr<VoXhUnV;|M0ve2Ig$yg^NU7v2%+Hn)e8bouLUH8`aEfg0E46BOGpdvI%FV>!IpsGmV!0fO}ff@#NPKPm#whd(ol4_ii;Yfnh5i>1teB)WC!`wo$tYY|Z|B_&(?22Bl{My#NTehSVTaiG}Va2bcqcTrgIl2fT~Ms10!ym@6oSfx6xs1zLe5RXI6-2}<|iGT@K@pebcFj=-GJByK|<}j6R=P{1T^ECX|wrIG(YI!SOKoF*HL?fMc+lwHO(E}~OJFI(v(tsh!L3tkzL=~ouonfhwj|j>Fh&)5Cw{{&BVJoOe0-NhoWjNlX)YhaB#}LeOKj&`tfS0`!wst4i$urMs;9Ee~#iI{4gA<Tu19hi5H<XD9J#G*hmZ=&JnIXPHXc0Bi6Hu|6%^@z6y$SBJWiAFRJTCPX5r5OyU&sAMKOT?Y^X3?O6XN=-OZ6e<vHB1by}i29e_pWHgdDwwuJqAf^-e%A-2aBRs0rC_%TI&mN27lOP)h*<6ay3h000O8002f^x|9$HU<&{MmLC8B6#xJL0000000000fB^si003ieZgypIbYEj=Wn*h_Z)t9HE^v8JO928D0~7!N00;m807hL554fJ1SpWc}Hvs@700000000000001h0h0>=0A^uxbZ}vGXkT_RFLQEZFLH2pF*aXoWpZw1Y;#|BGA?6qa8OGD0u%!j000080000+UBI(kF$5d{08(E7022TJ00000000000Du8+WdHzjaCR{^Ut@4}Uvp)0c4cxdaCuNm0Rj{Q6aWAK2mk;8MqLz_Dgh%J0075U000yK0000000000004jiq=5hca&UGrHeY68b98WFbZB38F)na<P)h*<6ay3h000O8002f^%+Cj?8&3cLyb%Ea6aWAK0000000000fC2fN0044ub}=?zW?^%5aA9<4Uv@GsaCuNm0Rj{Q6aWAK2mk;8MqQ;aJkggI003J}000#L0000000000004jiM(Y3oa&UGrHeYRZY;<XDZ)9a(b}=q+c~DCM0u%!j000080000+U7IXB4(18~052*401yBG00000000000Du7y0|5YZX=P+zV`F7sVrgM>E^v8JO9ci10000700#iu0RR9b4gmlF00"
_AGILLM_RUNTIME_MEMBERS = {'convert_checkpoint.py': '12d7789f2067424a081afffe6ccd76cca4cef0ccabaa800a6471db9ac1121a3c', 'fastpath_v2/src/rpv16_kernels_v2.cpp': '10d2147d5fe218b4d7fe5952b61942e68f877ce7269d0b0c99c861ee8512358c', 'rpv16_cpu_server.py': '6de1df5f1ca1294ba1d9c47c17e77e00f76b99c8582181cc106901eade96229b', 'rpv16_fastpath_v1.py': '76a418ee0380d4180b4d24eb704013ee540ea7bc8d639bad6138afe9f69ef17b', 'rpv16_fastpath_v2.py': 'b8c2c70ce64c5f1534adbf5520fa354023b3da44a29aa8f3a8f7153f4e1589dd', 'rpv16_multimode_v1.py': '1e806b8275ad18806971f2f91beb754aab2ac65be1a1920a2f959be0b1ef808d', 'tied_cce_bias.py': 'e1ad748629e19452bf2ba4f93d327014001b208bae0f293e5b2a8b164dbc95c8'}


def _ag_runtime(member_names=None):
    """Materialize only bundled, verified source files into a versioned user cache."""
    raw = _ag_b64.b85decode(_AGILLM_RUNTIME_B85)
    if _ag_hash.sha256(raw).hexdigest() != _AGILLM_RUNTIME_SHA256:
        raise RuntimeError("Embedded AGILLM runtime checksum mismatch")
    base = _ag_path.Path(_ag_os.environ.get("AGILLM_RUNTIME_CACHE", str(_ag_path.Path.home() / ".cache" / "agillm-singlefile")))
    root = base / _AGILLM_RUNTIME_SHA256[:24]
    root.mkdir(parents=True, exist_ok=True)
    selected = set(_AGILLM_RUNTIME_MEMBERS if member_names is None else member_names)
    if not selected.issubset(_AGILLM_RUNTIME_MEMBERS):
        raise ValueError("Unknown embedded runtime member")
    with _ag_zip.ZipFile(_ag_io.BytesIO(raw)) as archive:
        if set(archive.namelist()) != set(_AGILLM_RUNTIME_MEMBERS):
            raise RuntimeError("Unexpected embedded runtime archive members")
        for name in sorted(selected):
            destination = root / name
            if not destination.resolve().is_relative_to(root.resolve()):
                raise RuntimeError("Runtime cache path escaped its directory")
            data = archive.read(name)
            expected = _AGILLM_RUNTIME_MEMBERS[name]
            if _ag_hash.sha256(data).hexdigest() != expected:
                raise RuntimeError("Runtime member checksum mismatch: " + name)
            if destination.is_file() and _ag_hash.sha256(destination.read_bytes()).hexdigest() == expected:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = _ag_temp.mkstemp(prefix=".agillm-", dir=destination.parent)
            try:
                with _ag_os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    _ag_os.fsync(stream.fileno())
                _ag_os.replace(temporary, destination)
            finally:
                if _ag_os.path.exists(temporary):
                    _ag_os.unlink(temporary)
    return root


def _ag_inference_cli(argv):
    import argparse
    import importlib
    import contextlib
    import shutil
    import runpy
    parser = argparse.ArgumentParser(description="AGILLM RPV16 2B single-file training and inference")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("infer", "generate", "serve"):
        p = sub.add_parser(command)
        p.add_argument("--model-dir", required=True, help="Directory containing shared.pt and stage00.pt..stage15.pt (export-cpu creates it)")
        p.add_argument("--tokenizer", default="", help="Tokenizer JSON/bundle; defaults to tokenizer_bundle.json in --model-dir")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--kernel", choices=("portable", "avx512"), default="portable", help="Portable KV-cache path, or opt-in Linux AVX-512 compiled fast path")
        p.add_argument("--max-prompt", type=int, default=2048)
        if command == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=18620)
        else:
            p.add_argument("--prompt", required=True)
            p.add_argument("--mode", choices=("ar", "sat_fixed", "sat_variable", "sat-fixed", "sat-variable", "sat-var", "nat"), default="ar")
            p.add_argument("--max-new", type=int, default=32)
            p.add_argument("--temperature", type=float, default=0.0)
            p.add_argument("--top-k", type=int, default=0)
            p.add_argument("--top-p", type=float, default=1.0)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--ignore-eos", action="store_true")
            p.add_argument("--json", action="store_true", help="Print the full inference result, including token count and total generation throughput")
    export = sub.add_parser("export-cpu", help="Convert a trusted training checkpoint to the bundled CPU inference format")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--tokenizer", required=True)
    info = sub.add_parser("bundle-info", help="Print source provenance without loading PyTorch or model weights")
    args = parser.parse_args(argv)
    if args.command == "bundle-info":
        print(_ag_json.dumps({"schema": "agillm.singlefile.train-infer.v1", "model": "AGILLM RPV16 2B", "trainer_sha256_before_packaging": _AGILLM_ORIGINAL_TRAINER_SHA256, "runtime_sha256": _AGILLM_RUNTIME_SHA256, "members": _AGILLM_RUNTIME_MEMBERS, "weights_embedded": False}, indent=2))
        return
    runtime = _ag_runtime()
    if str(runtime) not in _ag_sys.path:
        _ag_sys.path.insert(0, str(runtime))
    if args.command == "export-cpu":
        source_checkpoint_path, output, tokenizer = _ag_path.Path(args.checkpoint).resolve(), _ag_path.Path(args.out).resolve(), _ag_path.Path(args.tokenizer).resolve()
        if not source_checkpoint_path.is_file() or not tokenizer.is_file():
            parser.error("The checkpoint and tokenizer must both be existing files")
        if output.exists() and any(output.iterdir()):
            parser.error("Use an empty output directory so an existing inference model is not overwritten")
        original_argv = _ag_sys.argv[:]
        try:
            _ag_sys.argv = [str(runtime / "convert_checkpoint.py"), "--source", str(source_checkpoint_path), "--out", str(output)]
            runpy.run_path(str(runtime / "convert_checkpoint.py"), run_name="__main__")
        finally:
            _ag_sys.argv = original_argv
        shutil.copyfile(tokenizer, output / "tokenizer_bundle.json")
        with (output / "SHA256SUMS").open("a", encoding="utf-8") as stream:
            stream.write(_ag_hash.sha256(tokenizer.read_bytes()).hexdigest() + "  tokenizer_bundle.json\n")
        return
    model_dir = _ag_path.Path(args.model_dir).resolve()
    tokenizer = _ag_path.Path(args.tokenizer).resolve() if args.tokenizer else model_dir / "tokenizer_bundle.json"
    if not (model_dir / "shared.pt").is_file() or not tokenizer.is_file():
        parser.error("Model files or tokenizer are missing. Supply an existing CPU export, or run export-cpu first")
    missing = [f"stage{i:02d}.pt" for i in range(16) if not (model_dir / f"stage{i:02d}.pt").is_file()]
    if missing:
        parser.error("Incomplete model export; missing " + ", ".join(missing))
    if args.threads < 1 or args.max_prompt < 1 or (args.command != "serve" and args.max_new < 1):
        parser.error("Thread and token counts must be positive")
    _ag_os.environ["RPV16_CPU_ROOT"] = str(model_dir)
    _ag_os.environ["RPV16_TOKENIZER"] = str(tokenizer)
    _ag_os.environ["RPV16_CPU_THREADS"] = str(args.threads)
    _ag_os.environ["RPV16_MAX_PROMPT"] = str(args.max_prompt)
    if args.command != "serve":
        _ag_os.environ["RPV16_MAX_NEW"] = str(args.max_new)
    try:
        meta = _ag_json.loads((model_dir / "manifest.json").read_text())
        _ag_os.environ["RPV16_SOURCE_CHECKPOINT_SHA"] = str(meta.get("source_sha256", "unknown"))
    except (OSError, ValueError):
        _ag_os.environ["RPV16_SOURCE_CHECKPOINT_SHA"] = "unknown"
    _ag_os.environ.setdefault("RPV16_FASTPATH_RECEIPTS", str(runtime / "receipts"))
    _ag_os.environ.setdefault("RPV16_FASTPATH_BUILD", str(runtime / "fastpath_v2" / "build"))
    if args.kernel == "avx512":
        cpuinfo = _ag_path.Path("/proc/cpuinfo")
        flags = cpuinfo.read_text().lower() if cpuinfo.is_file() else ""
        if not all(flag in flags for flag in ("avx512f", "avx512bw", "avx512vl", "avx512dq")):
            parser.error("The AVX-512 path requires a verified compatible Linux CPU. Use --kernel portable on this machine")
    with contextlib.redirect_stdout(_ag_sys.stderr):
        srv = importlib.import_module("rpv16_cpu_server")
        fastpath = importlib.import_module("rpv16_fastpath_v2" if args.kernel == "avx512" else "rpv16_fastpath_v1")
        fastpath.install(srv.__dict__)
    if args.command == "serve":
        import uvicorn
        uvicorn.run(srv.app, host=args.host, port=args.port, log_level="info")
        return
    import torch
    torch.manual_seed(args.seed)
    with contextlib.redirect_stdout(_ag_sys.stderr):
        engine = srv.engine()
    body = {"prompt": args.prompt, "mode": args.mode, "max_new": args.max_new, "temperature": args.temperature, "greedy": args.temperature <= 0, "top_k": args.top_k, "top_p": args.top_p, "ignore_eos": args.ignore_eos}
    import time
    import uuid
    result = None
    with torch.inference_mode():
        for event in engine.events(body, str(uuid.uuid4()), time.perf_counter()):
            if event.get("event") == "error":
                raise RuntimeError(str(event.get("error", "Inference failed")))
            if event.get("event") == "done":
                result = event
    if result is None:
        raise RuntimeError("Inference completed without a final result")
    if args.json:
        print(_ag_json.dumps(result, ensure_ascii=False))
    else:
        print(result.get("completion", result.get("text", "")))
        print(_ag_json.dumps(result.get("stats", {}), ensure_ascii=False), file=_ag_sys.stderr)


def _ag_singlefile_entry():
    if __name__ != "__main__":
        return
    argv = _ag_sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("AGILLM RPV16 2B: single-file training and inference\n\nCommands:\n  train             Original live GPU trainer; use train --help\n  profile           Original hardware profile\n  dataset-probe     Original dataset probe\n  infer / generate  CPU inference: AR, SAT fixed, learned SAT variable, NAT\n  serve             Serve the bundled CPU inference API\n  export-cpu        Convert a trusted training checkpoint for CPU inference\n  bundle-info       Embedded source checksums and provenance\n\nRun a command with --help for its options. Weights, tokenizer and third-party Python libraries are separate from this script.")
        raise SystemExit(0)
    if argv[0] in ("infer", "generate", "serve", "export-cpu", "bundle-info"):
        try:
            _ag_inference_cli(argv)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            print("AGILLM: " + type(exc).__name__ + ": " + str(exc), file=_ag_sys.stderr)
            raise SystemExit(1) from None
        raise SystemExit(0)


_ag_singlefile_entry()
# Keep existing local training modules higher-priority on the live box; a copied
# script resolves the same helper from its verified embedded cache instead.
_ag_training_runtime = _ag_runtime(["tied_cce_bias.py"])
if str(_ag_training_runtime) not in _ag_sys.path:
    _ag_sys.path.append(str(_ag_training_runtime))
# END AGILLM_2B_TRAIN_INFER_20261006

# AGILLM RPV16 standalone bundle v2: helper Python and Triton code embedded in this file.
"""Permutation-exact MoE scatter plus residual, with an unchanged gather VJP.

No routing, expert row order, quantization, or matrix multiplication changes.
The caller guarantees order is a permutation, as produced by stable argsort.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _sf_moe_scatter_add_kernel(X, Order, Out, E0, E1, E2, E3, E4, E5,
                 G:tl.constexpr, D:tl.constexpr, BLOCK:tl.constexpr):
    expert = tl.program_id(1)
    o = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    valid = o < G*D
    row = o // D
    col = o % D
    dest = tl.load(Order+expert*G+row, valid, other=0)
    if expert == 0:
        z=tl.load(E0+o,valid,other=0)
    elif expert == 1:
        z=tl.load(E1+o,valid,other=0)
    elif expert == 2:
        z=tl.load(E2+o,valid,other=0)
    elif expert == 3:
        z=tl.load(E3+o,valid,other=0)
    elif expert == 4:
        z=tl.load(E4+o,valid,other=0)
    else:
        z=tl.load(E5+o,valid,other=0)
    off=dest*D+col
    x=tl.load(X+off,valid,other=0)
    tl.store(Out+off,x.to(tl.float32)+z.to(tl.float32),valid)

class _SFResidualScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, order, groups, *experts):
        if len(experts)!=6:
            raise ValueError('exactly six expert tensors required')
        g=int(groups);d=int(residual.shape[-1])
        if residual.numel()!=6*g*d or not residual.is_contiguous():
            raise ValueError('invalid residual shape or strides')
        if not residual.is_cuda or residual.dtype not in (torch.bfloat16,torch.float16,torch.float32):
            raise ValueError('native kernel requires CUDA and fp32/fp16/bf16')
        if order.dtype!=torch.long or order.shape!=(6*g,) or not order.is_contiguous() or order.device!=residual.device:
            raise ValueError('invalid order metadata')
        if any(e.shape!=(g,d) or e.dtype!=residual.dtype or e.device!=residual.device or not e.is_contiguous() for e in experts):
            raise ValueError('invalid expert metadata')
        out=torch.empty_like(residual)
        _sf_moe_scatter_add_kernel[(triton.cdiv(g*d,1024),6)](residual,order,out,*experts,g,d,1024,num_warps=4)
        ctx.save_for_backward(order)
        ctx.groups=g;ctx.d=d
        return out
    @staticmethod
    def backward(ctx, grad):
        (order,)=ctx.saved_tensors
        # Exactly the original scatter backward. No new reduction is introduced.
        gathered=grad.reshape(-1,ctx.d).index_select(0,order)
        return (grad,None,None)+tuple(gathered.narrow(0,e*ctx.groups,ctx.groups) for e in range(6))

def _sf_moe_fused(residual,order,groups,*experts):
    return _SFResidualScatter.apply(residual,order,groups,*experts)

def _sf_moe_reference(residual,order,groups,*experts):
    flat=torch.empty_like(residual.reshape(-1,residual.shape[-1]))
    for e,z in enumerate(experts):
        flat.index_copy_(0,order.narrow(0,e*groups,groups),z)
    return residual+flat.view_as(residual)

"""Fused O(B*T*D) partner-key corrections for exact SAT attention."""
import torch
import triton
import triton.language as tl

@triton.jit
def _sf_sat_merge_kernel(Q,K,V,Y0,L0,F,Y,LT,P,
           T:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
           FS0:tl.constexpr,FS1:tl.constexpr,SCALE:tl.constexpr,
           BD:tl.constexpr):
    row=tl.program_id(0)
    h=row%H;i=(row//H)%T;b=row//(H*T);kh=h//(H//HK)
    d=tl.arange(0,BD)
    qoff=((b*T+i)*H+h)*D+d
    koff=((b*T+i+1)*HK+kh)*D+d
    active=(i+1<T)&tl.load(F+b*FS0+i*FS1)
    q=tl.load(Q+qoff,d<D,other=0).to(tl.float32)
    kp=tl.load(K+koff,(d<D)&(i+1<T),other=0).to(tl.float32)
    vp=tl.load(V+koff,(d<D)&(i+1<T),other=0).to(tl.float32)
    score=tl.sum(q*kp,axis=0)*SCALE
    score=tl.where(active,score,-float('inf'))
    lseoff=(b*H+h)*T+i
    l0=tl.load(L0+lseoff).to(tl.float32)
    mx=tl.maximum(l0,score)
    lt=mx+tl.log(tl.exp(l0-mx)+tl.exp(score-mx))
    pp=tl.exp(score-lt);beta=tl.exp(l0-lt)
    y0=tl.load(Y0+qoff,d<D,other=0).to(tl.float32)
    tl.store(Y+qoff,beta*y0+pp*vp,d<D)
    tl.store(LT+lseoff,lt)
    tl.store(P+row,pp)

@triton.jit
def _sf_sat_add_grads_kernel(Q,K,V,Y,G,P,DQ,DK,DV,
               T:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
               SCALE:tl.constexpr,BR:tl.constexpr,BD:tl.constexpr):
    row=tl.program_id(0)
    kh=row%HK;i=(row//HK)%T;b=row//(HK*T)
    r=tl.arange(0,BR);d=tl.arange(0,BD)
    R:tl.constexpr=H//HK
    hh=kh*R+r
    off=((b*T+i)*H+hh[:,None])*D+d[None,:]
    kv=((b*T+i)*HK+kh)*D+d
    kn=((b*T+i+1)*HK+kh)*D+d
    mask=(r[:,None]<R)&(d[None,:]<D)
    gg=tl.load(G+off,mask,other=0).to(tl.float32)
    yy=tl.load(Y+off,mask,other=0).to(tl.float32)
    pp=tl.load(P+(b*T+i)*H+hh,r<R,other=0).to(tl.float32)
    vn=tl.load(V+kn,(d<D)&(i+1<T),other=0).to(tl.float32)
    knval=tl.load(K+kn,(d<D)&(i+1<T),other=0).to(tl.float32)
    ds=pp*tl.sum(gg*(vn[None,:]-yy),axis=1)
    dq=tl.load(DQ+off,mask,other=0).to(tl.float32)
    tl.store(DQ+off,dq+ds[:,None]*knval[None,:]*SCALE,mask)
    # Incoming partner contribution to K_i/V_i is from query i-1.
    prev=((b*T+i-1)*H+hh[:,None])*D+d[None,:]
    pmask=mask&(i>0)
    gp=tl.load(G+prev,pmask,other=0).to(tl.float32)
    yp=tl.load(Y+prev,pmask,other=0).to(tl.float32)
    qp=tl.load(Q+prev,pmask,other=0).to(tl.float32)
    pprev=tl.load(P+(b*T+i-1)*H+hh,(r<R)&(i>0),other=0).to(tl.float32)
    vc=tl.load(V+kv,d<D,other=0).to(tl.float32)
    dsp=pprev*tl.sum(gp*(vc[None,:]-yp),axis=1)
    dk=tl.load(DK+kv,d<D,other=0).to(tl.float32)
    dv=tl.load(DV+kv,d<D,other=0).to(tl.float32)
    tl.store(DK+kv,dk+tl.sum(dsp[:,None]*qp,axis=0)*SCALE,d<D)
    tl.store(DV+kv,dv+tl.sum(pprev[:,None]*gp,axis=0),d<D)

def _sf_sat_merge_partner(q,k,v,base,lse,first2,scale):
    b,t,h,d=q.shape;hk=k.shape[2]
    out=torch.empty_like(base);total=torch.empty_like(lse)
    p=torch.empty((b,t,h),dtype=torch.float32,device=q.device)
    _sf_sat_merge_kernel[(b*t*h,)](q,k,v,base,lse,first2,out,total,p,
                    t,h,hk,d,first2.stride(0),first2.stride(1),scale,
                    triton.next_power_of_2(d),num_warps=4)
    return out,total,p

def _sf_sat_add_partner_grads_(g,q,k,v,out,p,dq,dk,dv,scale):
    b,t,h,d=q.shape;hk=k.shape[2]
    _sf_sat_add_grads_kernel[(b*t*hk,)](q,k,v,out,g,p,dq,dk,dv,t,h,hk,d,scale,
                         triton.next_power_of_2(h//hk),triton.next_power_of_2(d),num_warps=4)


import sys as _sf_sys, types as _sf_types, hashlib as _sf_hashlib

def _sf_install(name, source, *, package=False, injected=None):
    mod = _sf_types.ModuleType(name)
    mod.__file__ = "<singlefile:%s>" % name
    mod.__package__ = name if package else name.rpartition(".")[0]
    if package:
        mod.__path__ = []
    if injected:
        mod.__dict__.update(injected)
    _sf_sys.modules[name] = mod
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)
    return mod


_sf_install('sg_bnb_streamstep_c152b07f', '"""Instance-local BNB completion-barrier coalescing, not an optimizer change.\n\nOnly checkpoint-restored, already initialized ordinary CUDA state is eligible.\nManaged/paged state, new state, hooks, mixed devices, and unsupported versions\nuse the original method. Arithmetic, iteration and parameter-group order stay\nin bitsandbytes.update_step. A completion barrier remains at EVERY bank return.\n"""\nfrom __future__ import annotations\nimport json\nimport time\nimport types\nfrom pathlib import Path\nimport torch\nimport bitsandbytes as bnb\nfrom bitsandbytes.utils import sync_gpu\nfrom torch.optim import optimizer as _torch_optimizer\n\n_STOCK_STEP = bnb.optim.optimizer.Optimizer8bit.step\n_PROFILE_STEP_CODE = _torch_optimizer.Optimizer.profile_hook_step(_STOCK_STEP).__code__\n\nPOLICY = Path(\'/workspace/astra_dual_optimizer_20260924/policy.json\')\n_STATS = {}\n_ENABLED = False\n_REVISION = None\n_BARRIER_EVERY = 4\n\ndef begin_step():\n    global _STATS, _ENABLED, _REVISION, _BARRIER_EVERY\n    reason = None\n    try:\n        c = json.loads(POLICY.read_text())\n        assert type(c.get(\'enabled\')) is bool\n        active = c[\'enabled\']\n        if active and c.get(\'state\') != \'promoted\':\n            assert type(c.get(\'lease_until\')) in (int, float)\n            if time.time() >= c[\'lease_until\']:\n                active = False\n                reason = \'canary_lease_expired\'\n        _REVISION = c.get(\'revision\')\n        be = c.get(\'barrier_every\', 4)\n        if active:\n            assert type(be) is int and 1 <= be <= 512\n        _BARRIER_EVERY = int(be) if active else 4\n    except (OSError, ValueError, AssertionError, TypeError):\n        active = False\n        reason = \'missing_or_invalid_policy\'\n    _ENABLED = active\n    _STATS = dict(enabled=active, revision=_REVISION, policy_fallback=reason,\n                  bank_calls=0, fast_calls=0, fallback_calls=0, params_updated=0,\n                  completion_barriers=0, optimizer_wall_s=0.0, barrier_every=_BARRIER_EVERY, fallback_reasons={})\n\ndef telemetry():\n    return dict(_STATS, fallback_reasons=dict(_STATS.get(\'fallback_reasons\', {})))\n\ndef _stock_step_method(method):\n    # Accept the exact BNB method or the known PyTorch profiling wrapper only.\n    # Arbitrary @wraps(stock) custom methods must retain their own execution.\n    fn = getattr(method, \'__func__\', method)\n    return fn is _STOCK_STEP or (\n        getattr(fn, \'__code__\', None) is _PROFILE_STEP_CODE and\n        getattr(fn, \'__wrapped__\', None) is _STOCK_STEP)\n\ndef _eligibility(opt):\n    for name in (\'_global_optimizer_pre_hooks\', \'_global_optimizer_post_hooks\'):\n        hooks = getattr(_torch_optimizer, name, None)\n        if hooks is None: return \'unsupported_global_hook_api\'\n        if hooks: return \'global_optimizer_hooks\'\n    if not _stock_step_method(getattr(opt, \'_sg_streamstep_original\', None)):\n        return \'custom_optimizer_step\'\n    if getattr(bnb, \'__version__\', None) != \'0.50.2\': return \'unvalidated_bnb_version\'\n    if not opt.initialized: return \'uninitialized_bank\'\n    if getattr(opt, \'_optimizer_step_pre_hooks\', {}) or getattr(opt, \'_optimizer_step_post_hooks\', {}): return \'optimizer_hooks\'\n    devices = set()\n    active = []\n    ptrs = set()\n    for gi, group in enumerate(opt.param_groups):\n        for pi, p in enumerate(group[\'params\']):\n            if p.grad is None: continue\n            if not p.is_cuda or p.grad.device != p.device: return \'noncuda_or_mixed_device\'\n            devices.add(p.device)\n            if p.data_ptr() in ptrs: return \'aliased_parameter\'\n            ptrs.add(p.data_ptr())\n            st = opt.state.get(p)\n            if not st or \'state1\' not in st or \'state2\' not in st: return \'new_or_incomplete_state\'\n            for v in st.values():\n                if torch.is_tensor(v):\n                    if getattr(v, \'is_paged\', False): return \'managed_state\'\n                    if not v.is_cuda or v.device != p.device: return \'nonresident_state\'\n            active.append((gi, pi, group, p))\n    if len(devices) != 1: return \'empty_or_mixed_device_bank\'\n    return active\n\ndef install(opt):\n    if getattr(opt, \'_sg_streamstep_installed\', False): return False\n    if type(opt) is not bnb.optim.PagedAdamW8bit: return False\n    original = opt.step\n    if not _stock_step_method(original): return False\n    opt._sg_streamstep_original = original\n    @torch.no_grad()\n    def step(self, closure=None):\n        start = time.perf_counter()\n        _STATS[\'bank_calls\'] = _STATS.get(\'bank_calls\', 0) + 1\n        eligible = _eligibility(self) if _ENABLED else \'policy_disabled\'\n        if isinstance(eligible, str):\n            _STATS[\'fallback_calls\'] = _STATS.get(\'fallback_calls\', 0) + 1\n            reasons = _STATS.setdefault(\'fallback_reasons\', {})\n            reasons[eligible] = reasons.get(eligible, 0) + 1\n            n = sum(p.grad is not None for g in self.param_groups for p in g[\'params\'])\n            loss = original(closure)\n            _STATS[\'params_updated\'] = _STATS.get(\'params_updated\', 0) + n\n            _STATS[\'completion_barriers\'] = _STATS.get(\'completion_barriers\', 0) + n + int(bool(self.is_paged))\n        else:\n            # A closure could change gradient presence or device. Use stock for it.\n            if closure is not None:\n                loss = original(closure)\n                _STATS[\'fallback_calls\'] = _STATS.get(\'fallback_calls\', 0) + 1\n                reasons = _STATS.setdefault(\'fallback_reasons\', {})\n                reasons[\'closure\'] = reasons.get(\'closure\', 0) + 1\n            else:\n                loss = None\n                barriers = 0\n                for idx, (gi, pi, group, p) in enumerate(eligible, 1):\n                    self.prefetch_state(p)  # exact stock call; no-op for audited resident state\n                    self.update_step(group, p, gi, pi)\n                    if idx % _BARRIER_EVERY == 0:\n                        sync_gpu(p)\n                        barriers += 1\n                if len(eligible) % _BARRIER_EVERY:\n                    sync_gpu(eligible[-1][3])  # preserve bank-completion guarantee\n                    barriers += 1\n                _STATS[\'fast_calls\'] = _STATS.get(\'fast_calls\', 0) + 1\n                _STATS[\'params_updated\'] = _STATS.get(\'params_updated\', 0) + len(eligible)\n                _STATS[\'completion_barriers\'] = _STATS.get(\'completion_barriers\', 0) + barriers\n        _STATS[\'optimizer_wall_s\'] = _STATS.get(\'optimizer_wall_s\', 0.0) + time.perf_counter() - start\n        return loss\n    opt.step = types.MethodType(step, opt)\n    opt._sg_streamstep_installed = True\n    return True\n')
_sf_install('sg_moe_residual_d803c31d', '"""Reversible production dispatch for permutation-exact residual scatter."""\nfrom __future__ import annotations\nfrom pathlib import Path\nimport hashlib,json,os,sys,time,types\nimport torch\nROOT=Path(\'/workspace/agillm-gb10-1pf-selftest/sg_moe_residual_d803c31d\')\nKERNEL_SHA="d803c31d9a12fd6b4ddc09abc9660332140e46c6722b0b72af413f4d9528e9c7"\n_backend=None\n_checked=float(\'-inf\');_mode=\'reference\';_error=None\n_counts={\'fused_calls\':0,\'reference_calls\':0};_seen=set()\n\ndef policy():\n global _checked,_mode,_error\n now=time.monotonic()\n if now-_checked<5:return _mode\n _checked=now;_error=None\n try:\n  mode=os.environ.get(\'AGILLM_MOE_RESIDUAL\')\n  if mode is None:mode=json.loads((ROOT/\'policy.json\').read_text()).get(\'backend\',\'reference\')\n  if mode not in (\'auto\',\'reference\'):raise ValueError(\'unknown backend\')\n  _mode=mode\n except (OSError,ValueError,TypeError,AttributeError) as e:\n  _mode=\'reference\';_error=str(e)[:120]\n return _mode\n\ndef eligible(x,order,g,es):\n return (x.is_cuda and x.dtype==torch.bfloat16 and x.is_contiguous()\n  and x.shape==(24,2048,1280) and g==8192 and len(es)==6\n  and order.shape==(49152,) and order.dtype==torch.long and order.is_contiguous()\n  and order.device==x.device and all(z.shape==(8192,1280) and z.dtype==x.dtype\n  and z.device==x.device and z.is_contiguous() for z in es))\n\ndef load():\n global _backend\n if _backend is None:_backend=_SF_MOE_FUSED\n return _backend\n\n\ndef scatter_add(x,order,groups,experts,original,emit=None):\n mode=policy();use=mode==\'auto\' and eligible(x,order,groups,experts)\n if use:out=load()(x,order,groups,*experts);_counts[\'fused_calls\']+=1\n else:out=x+original(order,groups,*experts).view_as(x);_counts[\'reference_calls\']+=1\n key=(use,torch.is_grad_enabled(),_error)\n if key not in _seen or (use and _counts[\'fused_calls\']%256==0):\n  _seen.add(key)\n  record=dict(event=\'moe_residual_backend_active\',schema=\'agillm.moe-residual.production.v1\',\n   backend=\'fused_scatter_residual\' if use else \'reference\',shape=list(x.shape),\n   grad_enabled=torch.is_grad_enabled(),kernel_sha256=KERNEL_SHA,policy=mode,policy_error=_error,\n   eliminated_intermediate_bytes=x.numel()*x.element_size() if use else 0,**_counts)\n  if emit:emit(record)\n  else:print(json.dumps(record,allow_nan=False),flush=True)\n return out\n', package=True, injected={'_SF_MOE_FUSED': _sf_moe_fused})
_sf_sat_tri = _sf_types.ModuleType('sg_sat_partner_0ac2c32d.sat_partner_triton')
_sf_sat_tri.__file__ = __file__
_sf_sat_tri.__package__ = 'sg_sat_partner_0ac2c32d'
_sf_sat_tri.merge_partner = _sf_sat_merge_partner
_sf_sat_tri.add_partner_grads_ = _sf_sat_add_partner_grads_
_sf_sys.modules['sg_sat_partner_0ac2c32d.sat_partner_triton'] = _sf_sat_tri
_sf_sat_attn = _sf_install('sg_sat_partner_0ac2c32d.sat_partner_attention', '"""Exact SAT clean-context attention using one causal pass plus a partner key.\n\nExperimental AGILLM backend. No noisy tensors are used as attention K/V.\nFor a first-of-pair row i the allowed clean keys are [i-window, i+1];\nother rows use [i-window, i]. Negative window means an unbounded left edge.\n\nThe native backend targets the installed FlashAttention 2.x low-level API.\nThe CPU backend is an executable mathematical reference, NOT a speed backend.\nOnly dropout=0, no ALiBi/softcap, equal Q/K sequence length are supported.\n"""\nfrom __future__ import annotations\n\nimport math\nfrom functools import lru_cache\nfrom typing import Literal\n\nimport torch\nfrom torch.autograd.function import once_differentiable\n\nSCHEMA = "agillm.sat-partner-attention.v1"\nBackend = Literal["auto", "reference", "flash", "flash_fused"]\n\n\ndef _validate(q, k, v, first2, window):\n    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:\n        raise ValueError("q must be [B,T,Hq,D]; k and v must be [B,T,Hkv,D]")\n    b, t, h, d = q.shape\n    if min(b, t, h, d) < 1 or k.shape[:2] != (b, t) or k.shape[-1] != d:\n        raise ValueError("incompatible or empty attention dimensions")\n    if k.shape[2] < 1 or h % k.shape[2]:\n        raise ValueError("Hq must be a positive multiple of Hkv")\n    if not q.is_floating_point() or k.dtype != q.dtype or v.dtype != q.dtype:\n        raise ValueError("q, k, v must have the same floating-point dtype")\n    if k.device != q.device or v.device != q.device or first2.device != q.device:\n        raise ValueError("q, k, v and first2 must share a device")\n    if type(window) is not int or window < -1:\n        raise ValueError("window must be -1 or a nonnegative integer")\n    if first2.dtype != torch.bool or first2.shape not in ((t,), (b, t)):\n        raise ValueError("first2 must be bool[T] or bool[B,T]")\n    f = first2[None, :].expand(b, -1) if first2.ndim == 1 else first2\n    if bool(f[:, -1].any()) or (t > 1 and bool((f[:, :-1] & f[:, 1:]).any())):\n        raise ValueError("invalid overlapping or out-of-bounds SAT pairs")\n    return f\n\n\ndef _acc_dtype(q):\n    return torch.float64 if q.dtype == torch.float64 else torch.float32\n\n\ndef _base_mask(t, window, device):\n    i = torch.arange(t, device=device)[:, None]\n    j = torch.arange(t, device=device)[None, :]\n    return (j <= i) & ((j >= i-window) if window >= 0 else True)\n\n\ndef _expanded_kv(k, v, hq, dtype):\n    r = hq // k.shape[2]\n    return (k.to(dtype).transpose(1, 2).repeat_interleave(r, dim=1),\n            v.to(dtype).transpose(1, 2).repeat_interleave(r, dim=1))\n\n\ndef _reference_base_forward(q, k, v, window, scale):\n    dt = _acc_dtype(q)\n    qq = q.to(dt).transpose(1, 2)\n    kk, vv = _expanded_kv(k, v, q.shape[2], dt)\n    s = (qq @ kk.transpose(-1, -2)) * scale\n    s = s.masked_fill(~_base_mask(q.shape[1], window, q.device), -torch.inf)\n    lse = torch.logsumexp(s, dim=-1)\n    out = (torch.softmax(s, dim=-1) @ vv).transpose(1, 2).contiguous().to(q.dtype)\n    return out, lse, torch.empty(0, dtype=torch.uint8, device=q.device)\n\n\n@lru_cache(maxsize=1)\ndef _flash_api():\n    # Do not silently fall back to a quadratic reference on a production GPU.\n    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward\n    return _flash_attn_forward, _flash_attn_backward\n\n\ndef _merge_partner(q, k, v, base_out, base_lse, first2, scale):\n    """Stable log-sum-exp composition; never materializes a T x T tensor."""\n    b, t, hq, d = q.shape\n    hk = k.shape[2]\n    r = hq // hk\n    dt = _acc_dtype(q)\n    qs = q[:, :-1].to(dt).reshape(b, t-1, hk, r, d)\n    ks = k[:, 1:].to(dt).unsqueeze(3)\n    extra_score = (qs * ks).sum(-1).reshape(b, t-1, hq) * scale\n    extra_score = extra_score.masked_fill(~first2[:, :-1, None], -torch.inf)\n    extra_score = torch.cat((extra_score, torch.full((b, 1, hq), -torch.inf,\n                            dtype=dt, device=q.device)), dim=1)\n    l0 = base_lse.transpose(1, 2).to(dt)\n    lt = torch.logaddexp(l0, extra_score)\n    p = torch.exp(extra_score-lt)\n    beta = torch.exp(l0-lt)\n    out = base_out.to(dt) * beta[..., None]\n    out[:, :-1] += (p[:, :-1].reshape(b, t-1, hk, r, 1) *\n                      v[:, 1:].to(dt).unsqueeze(3)).reshape(b, t-1, hq, d)\n    return out.to(q.dtype).contiguous(), lt.transpose(1, 2).contiguous(), p\n\n\ndef _reference_base_backward(g, q, k, v, out, total_lse, window, scale):\n    """Base-key softmax gradient with TOTAL normalizer and TOTAL output.\n\n    The base-key probabilities intentionally sum to <1 on pair-start rows.\n    Reusing the ordinary base output or its normalizer here would be wrong.\n    """\n    dt = _acc_dtype(q)\n    b, t, hq, d = q.shape\n    hk = k.shape[2]\n    qq, gg = q.to(dt).transpose(1, 2), g.to(dt).transpose(1, 2)\n    kk, vv = _expanded_kv(k, v, hq, dt)\n    scores = (qq @ kk.transpose(-1, -2))*scale\n    scores = scores.masked_fill(~_base_mask(t, window, q.device), -torch.inf)\n    p = torch.exp(scores-total_lse[..., None])\n    delta = (gg*out.to(dt).transpose(1, 2)).sum(-1, keepdim=True)\n    ds = p * (gg @ vv.transpose(-1, -2)-delta)\n    dq = (ds @ kk)*scale\n    dk = (ds.transpose(-1, -2) @ qq)*scale\n    dv = p.transpose(-1, -2) @ gg\n    dk = dk.reshape(b, hk, hq//hk, t, d).sum(2)\n    dv = dv.reshape(b, hk, hq//hk, t, d).sum(2)\n    return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2)\n\n\ndef _partner_backward(g, q, k, v, out, p, scale):\n    b, t, hq, d = q.shape\n    hk = k.shape[2]\n    r = hq//hk\n    dt = _acc_dtype(q)\n    gg = g[:, :-1].to(dt).reshape(b, t-1, hk, r, d)\n    yy = out[:, :-1].to(dt).reshape(b, t-1, hk, r, d)\n    pp = p[:, :-1].reshape(b, t-1, hk, r, 1)\n    vp, kp = v[:, 1:].to(dt).unsqueeze(3), k[:, 1:].to(dt).unsqueeze(3)\n    ds = pp * (gg * (vp-yy)).sum(-1, keepdim=True)\n    dq = (ds*kp*scale).reshape(b, t-1, hq, d)\n    dk = (ds*q[:, :-1].to(dt).reshape(b, t-1, hk, r, d)*scale).sum(3)\n    dv = (pp*gg).sum(3)\n    return dq, dk, dv\n\n\nclass _SATPartner(torch.autograd.Function):\n    @staticmethod\n    def forward(ctx, q, k, v, first2, window, scale, backend, deterministic):\n        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()\n        ctx.singleton = q.shape[1] == 1\n        if ctx.singleton:\n            # Softmax of one allowed key is exactly one; Q and K have zero gradient.\n            ctx.save_for_backward(q, k, v)\n            return v.repeat_interleave(q.shape[2]//k.shape[2], dim=2).contiguous()\n        if backend in ("flash", "flash_fused"):\n            fw, _ = _flash_api()\n            base, lse, _, rng = fw(q, k, v, 0.0, scale, True,\n                                  window, 0 if window >= 0 else -1, 0.0, None, False)\n        else:\n            base, lse, rng = _reference_base_forward(q, k, v, window, scale)\n        if backend == "flash_fused":\n            if __package__:\n                from .sat_partner_triton import merge_partner\n            else:\n                from sat_partner_triton import merge_partner\n            out, total_lse, p = merge_partner(q, k, v, base, lse, first2, scale)\n        else:\n            out, total_lse, p = _merge_partner(q, k, v, base, lse, first2, scale)\n        ctx.save_for_backward(q, k, v, out, total_lse, p, rng)\n        ctx.window, ctx.scale, ctx.backend = window, scale, backend\n        ctx.deterministic = deterministic\n        return out\n\n    @staticmethod\n    @once_differentiable\n    def backward(ctx, g):\n        if ctx.singleton:\n            q, k, v = ctx.saved_tensors\n            b, t, hk, d = k.shape\n            dv = g.to(_acc_dtype(q)).reshape(b, t, hk, q.shape[2]//hk, d).sum(3)\n            return torch.zeros_like(q), torch.zeros_like(k), dv.to(v.dtype), None, None, None, None, None\n        q, k, v, out, total_lse, p, rng = ctx.saved_tensors\n        g = g.contiguous()\n        if ctx.backend in ("flash", "flash_fused"):\n            _, bw = _flash_api()\n            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)\n            bw(g, q, k, v, out, total_lse.float(), dq, dk, dv, 0.0, ctx.scale,\n               True, ctx.window, 0 if ctx.window >= 0 else -1, 0.0, None,\n               ctx.deterministic, rng_state=rng)\n            if ctx.backend == "flash_fused":\n                if __package__:\n                    from .sat_partner_triton import add_partner_grads_\n                else:\n                    from sat_partner_triton import add_partner_grads_\n                add_partner_grads_(g,q,k,v,out,p,dq,dk,dv,ctx.scale)\n                return dq, dk, dv, None, None, None, None, None\n            # Accumulate the small correction in FP32, then cast once.\n            dt = _acc_dtype(q)\n            dq, dk, dv = dq.to(dt), dk.to(dt), dv.to(dt)\n        else:\n            dq, dk, dv = _reference_base_backward(g, q, k, v, out, total_lse,\n                                                ctx.window, ctx.scale)\n        dqp, dkp, dvp = _partner_backward(g, q, k, v, out, p, ctx.scale)\n        dq[:, :-1] += dqp\n        dk[:, 1:] += dkp\n        dv[:, 1:] += dvp\n        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None\n\n\ndef sat_partner_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,\n                          first2: torch.Tensor, window: int = -1,\n                          *, backend: Backend = "auto", deterministic: bool = True,\n                          softmax_scale: float | None = None, validate: bool = True):\n    """Clean-context SAT attention with exact mask and first-order gradients.\n\n    `validate=False` is for a caller that already validated the partition. It\n    avoids device-to-host validation synchronizations, NOT a change in semantics.\n    Existing learned per-example 1/2-token choices are passed through unchanged.\n    """\n    if validate:\n        f = _validate(q, k, v, first2, window)\n    else:\n        f = first2[None, :].expand(q.shape[0], -1) if first2.ndim == 1 else first2\n    if backend not in ("auto", "reference", "flash", "flash_fused"):\n        raise ValueError("backend must be auto, reference, flash, or flash_fused")\n    if backend == "auto":\n        backend = "flash" if q.is_cuda else "reference"\n    if backend in ("flash", "flash_fused") and (not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16)\n                              or q.shape[-1] % 8):\n        raise ValueError("flash backend requires CUDA, FP16/BF16, head_dim divisible by 8")\n    scale = q.shape[-1]**-0.5 if softmax_scale is None else float(softmax_scale)\n    if not math.isfinite(scale) or scale <= 0:\n        raise ValueError("softmax_scale must be positive and finite")\n    return _SATPartner.apply(q, k, v, f, window, scale, backend, deterministic)\n\n\ndef dense_sat_reference(q, k, v, first2, window=-1, softmax_scale=None):\n    """Independent dense oracle using one explicit full allowed-key mask."""\n    f = _validate(q, k, v, first2, window)\n    t, hq = q.shape[1:3]\n    dt = _acc_dtype(q)\n    scale = q.shape[-1]**-0.5 if softmax_scale is None else float(softmax_scale)\n    qq = q.to(dt).transpose(1, 2)\n    kk, vv = _expanded_kv(k, v, hq, dt)\n    i = torch.arange(t, device=q.device)[None, :, None]\n    j = torch.arange(t, device=q.device)[None, None, :]\n    mask = (j <= i+f[:, :, None].long())\n    if window >= 0:\n        mask = mask & (j >= i-window)\n    score = ((qq @ kk.transpose(-1, -2))*scale).masked_fill(~mask[:, None], -torch.inf)\n    return (torch.softmax(score, dim=-1) @ vv).transpose(1, 2).contiguous().to(q.dtype)\n')
_sf_install('sg_sat_partner_0ac2c32d', '"""Production SAT partner dispatch with pinned code and reversible policy.\n\nOnly replaces the two-pass attention implementation. It does not change SAT\npartitions, its learned stride policy, objectives, optimizer or token accounting.\n"""\nfrom __future__ import annotations\nimport hashlib\nimport json\nimport os\nfrom pathlib import Path\nimport sys\nimport time\nimport types\nfrom functools import lru_cache\nimport torch\n\nSCHEMA = \'agillm.sat-partner.production.v1\'\n_ROOT = Path(\'/workspace/agillm-gb10-1pf-selftest/sg_sat_partner_0ac2c32d\')\n_DEFAULT_POLICY = _ROOT / \'policy.json\'\n_HASHES = {\n    \'sat_partner_attention\': \'0ac2c32d8d68d58612942f8cf33e3b9cfb5225507d028ac3c0fe9ef167630101\',\n    \'sat_partner_triton\': \'930e8d214a7d30d3b8c1b0a7aa46c63d336884c0dcf555a266502f28cede942b\',\n}\n_backend = None\n_policy_checked = float(\'-inf\')\n_policy_mode = \'auto\'\n_policy_error = None\n_counts = {\'partner_dispatches\': 0, \'two_pass_dispatches\': 0}\n_seen = set()\n\n\ndef _read_policy(now=None):\n    """Poll once per five seconds. Malformed policy selects the safe old path."""\n    global _policy_checked, _policy_mode, _policy_error\n    t = time.monotonic() if now is None else float(now)\n    if t - _policy_checked < 5.0:\n        return _policy_mode\n    _policy_checked = t\n    _policy_error = None\n    try:\n        env = os.environ.get(\'AGILLM_SAT_PARTNER\')\n        if env is not None:\n            mode = env\n        else:\n            path = Path(os.environ.get(\'AGILLM_SAT_PARTNER_POLICY\', str(_DEFAULT_POLICY)))\n            mode = json.loads(path.read_text()).get(\'backend\', \'two_pass\')\n        if mode not in (\'auto\', \'two_pass\'):\n            raise ValueError(\'backend must be auto or two_pass\')\n        _policy_mode = mode\n    except (OSError, ValueError, TypeError, AttributeError) as exc:\n        _policy_mode = \'two_pass\'\n        _policy_error = type(exc).__name__ + \': \' + str(exc)[:160]\n    return _policy_mode\n\n\n@lru_cache(maxsize=4)\ndef _capability(device):\n    return tuple(torch.cuda.get_device_capability(device))\n\n\ndef eligible(q, k, v, first2, window):\n    """Metadata-only production shape gate, without GPU-to-CPU scalar reads."""\n    return (q.is_cuda and q.dtype == torch.bfloat16\n            and q.shape == (24, 2048, 20, 64)\n            and k.shape == (24, 2048, 5, 64) and v.shape == k.shape\n            and k.dtype == q.dtype and v.dtype == q.dtype\n            and k.device == q.device and v.device == q.device\n            and first2.dtype == torch.bool and first2.shape == (24, 2048)\n            and first2.device == q.device\n            and window in (256, 512, 1024, -1)\n            and _capability(q.device) == (12, 1))\n\n\ndef _load_backend():\n    global _backend\n    if _backend is None:\n        _backend = _SF_SAT_BACKEND\n    return _backend\n\n\ndef snapshot():\n    return dict(_counts, backend_policy=_policy_mode, policy_error=_policy_error,\n                implementation_sha256=_HASHES[\'sat_partner_attention\'])\n\n\ndef dispatch(q, k, v, window, first2, original, emit=None):\n    """Use one causal pass plus the exact partner correction on tested shapes."""\n    w = -1 if window is None else int(window)\n    mode = _read_policy()\n    use_partner = mode == \'auto\' and eligible(q, k, v, first2, w)\n    if use_partner:\n        result = _load_backend()(q, k, v, first2, w,\n                                 backend=\'flash_fused\', validate=False)\n        _counts[\'partner_dispatches\'] += 1\n    else:\n        result = original()\n        _counts[\'two_pass_dispatches\'] += 1\n    key = (use_partner, w, bool(torch.is_grad_enabled()), _policy_error)\n    if key not in _seen or (use_partner and _counts[\'partner_dispatches\'] % 64 == 0):\n        _seen.add(key)\n        record = dict(event=\'sat_partner_backend_active\', schema=SCHEMA,\n                      selected_backend=\'one_pass_partner\' if use_partner else \'two_pass\',\n                      window=w, shape=list(q.shape), grad_enabled=bool(torch.is_grad_enabled()),\n                      source=\'<singlefile>\', **snapshot())\n        # These are dispatch counts, not completed optimizer steps or tokens.\n        if emit is None:\n            print(json.dumps(record, allow_nan=False), flush=True)\n        else:\n            emit(record)\n    return result\n', package=True, injected={'_SF_SAT_BACKEND': _sf_sat_attn.sat_partner_attention})

# === ORIGINAL TRAINER SOURCE ===

import argparse
import copy
import ctypes
import json
import math
import os
# Execution-runtime selection only, before torch or CUDA initialization.
_SG_MPS_POLICY_PATH = "/workspace/astra_dual_mps_20260924/runtime_policy.json"
try:
    with open(_SG_MPS_POLICY_PATH) as _sg_mps_f:
        _sg_mps_config = json.load(_sg_mps_f)
    _sg_mps_enable = _sg_mps_config.get("enabled") is True
except (OSError, ValueError, TypeError):
    _sg_mps_enable = False
os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/workspace/astra_dual_mps_20260924/production_pipe" if _sg_mps_enable else ""
print(json.dumps({"event":"cuda_process_sharing_selection","mps_requested":_sg_mps_enable,
                  "pipe":os.environ["CUDA_MPS_PIPE_DIRECTORY"]}),flush=True)
import queue
import random
import signal
import statistics
import shutil
import sys
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import OrderedDict, deque
from pathlib import Path

_PROC_T0 = time.time()  # v2 telemetry: process start, for restart-cost accounting

import torch
import torch.nn as nn
import torch.nn.functional as F

# Preserve fixed sparse connectivity across checkpoint/resume (2026-10-03).
_RPV_SPARSE_TOPOLOGY = _sf_install("rpv16_sparse_topology_20261003", '"""Persist RPV16\'s fixed paired 4-of-8 topology outside model.state_dict().\n\nOne nibble holds the four adjacent-pair selection bits for eight weights;\ntwo such groups fit in one byte. Legacy checkpoints remain loadable, with\nan explicit event that their original running topology cannot be recovered.\n"""\n\nimport math\n\nimport torch\n\n\nSCHEMA = "agillm.sparse-topology.paired4of8.v1"\nFORMAT = "adjacent-pairs-low-nibble-first"\nKEY = "sparse_topology"\n\n\ndef _shape(shape):\n    if (not isinstance(shape, (list, tuple)) or len(shape) != 2\n            or any(type(n) is not int or n <= 0 for n in shape)\n            or math.prod(shape) % 8):\n        raise ValueError("sparse topology requires a positive 2D shape divisible by eight")\n    return tuple(shape)\n\n\n@torch.no_grad()\ndef pack_mask(mask):\n    """Return an independent CPU uint8 tensor without changing the live mask."""\n    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:\n        raise ValueError("sparse topology mask must be a bool tensor")\n    _shape(tuple(mask.shape))\n    groups = mask.detach().contiguous().reshape(-1, 8)\n    pairs = groups[:, 0::2]\n    if not torch.equal(pairs, groups[:, 1::2]):\n        raise ValueError("sparse topology mask does not select adjacent pairs")\n    if not bool((pairs.sum(dim=1) == 2).all()):\n        raise ValueError("sparse topology mask must select exactly two pairs per group")\n    bits = pairs[:, 0].to(torch.uint8)\n    for shift in range(1, 4):\n        bits.bitwise_or_(pairs[:, shift].to(torch.uint8).bitwise_left_shift(shift))\n    packed = bits[0::2].clone()\n    packed[: bits.numel() // 2].bitwise_or_(bits[1::2].bitwise_left_shift(4))\n    return packed.cpu().clone()\n\n\n@torch.no_grad()\ndef unpack_mask(packed, shape, device="cpu"):\n    """Validate the canonical byte representation before allocating on device."""\n    shape = _shape(shape)\n    groups = math.prod(shape) // 8\n    if (not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8\n            or packed.ndim != 1 or packed.numel() != (groups + 1) // 2\n            or packed.device.type != "cpu" or packed.layout != torch.strided):\n        raise ValueError("sparse topology packed tensor has wrong shape, dtype, or device")\n    packed = packed.detach().contiguous()\n    if groups % 2 and int(packed[-1]) & 0xF0:\n        raise ValueError("sparse topology unused high nibble must be zero")\n    bits = torch.empty(groups, dtype=torch.uint8)\n    bits[0::2] = packed.bitwise_and(15)\n    bits[1::2] = packed[: groups // 2].bitwise_right_shift(4)\n    valid = (bits == 3) | (bits == 5) | (bits == 6) | (bits == 9) | (bits == 10) | (bits == 12)\n    if not bool(valid.all()):\n        raise ValueError("sparse topology packed group must contain exactly two selected pairs")\n    pairs = bits[:, None].bitwise_and(torch.tensor([1, 2, 4, 8], dtype=torch.uint8)).ne(0)\n    return pairs.repeat_interleave(2, dim=1).reshape(shape).to(device=device)\n\n\ndef _modules(model, sparse_type):\n    return {name: module for name, module in model.named_modules()\n            if isinstance(module, sparse_type)}\n\n\ndef _install_mask(module, mask):\n    module._fixed_mask = mask\n    for name in ("_cache_key", "_wq", "_wq_key", "_packed_w", "_sf_w", "_wg"):\n        setattr(module, name, None)\n    # Retain ownership of native contexts. SparseLinear.shadow() sees the empty\n    # cache key and repacks every existing context (or closes them under the\n    # non-inplace policy) before use. Dropping handles here would leak memory;\n    # closing them during validation would defeat all-or-nothing restoration.\n\n\n@torch.no_grad()\ndef export_sparse_topology(model, sparse_type):\n    entries = {}\n    initialized = 0\n    for name, module in _modules(model, sparse_type).items():\n        shape = _shape(tuple(module.weight.shape))\n        mask = module._fixed_mask\n        if mask is not None and tuple(mask.shape) != shape:\n            raise ValueError("sparse topology mask/weight shape mismatch: " + name)\n        entries[name] = {"shape": list(shape), "packed": None if mask is None else pack_mask(mask)}\n        initialized += mask is not None\n    return {"schema": SCHEMA, "format": FORMAT, "module_count": len(entries),\n            "initialized_count": initialized, "entries": entries}\n\n\n@torch.no_grad()\ndef restore_sparse_topology(model, sparse_type, checkpoint, emit=None):\n    """Restore all masks, or reject without modifying any module.\n\nCall immediately after loading model weights and before its first forward.\nAbsent legacy state preserves the old lazy-initialization behavior. Explicit\nnull or malformed state is not treated as a legacy checkpoint.\n    """\n    if not isinstance(checkpoint, dict):\n        raise ValueError("sparse topology checkpoint must be a dictionary")\n    modules = _modules(model, sparse_type)\n    if KEY not in checkpoint:\n        for module in modules.values():\n            _install_mask(module, None)\n        event = {"event": "sparse_topology_resume", "status": "legacy_missing",\n                 "exact_topology_restored": False, "module_count": len(modules),\n                 "detail": "Legacy checkpoint has no fixed masks; masks initialize from loaded weights."}\n        if emit is not None:\n            emit(event)\n        return event\n    blob = checkpoint[KEY]\n    keys = {"schema", "format", "module_count", "initialized_count", "entries"}\n    if (not isinstance(blob, dict) or set(blob) != keys or blob["schema"] != SCHEMA\n            or blob["format"] != FORMAT or type(blob["module_count"]) is not int\n            or type(blob["initialized_count"]) is not int):\n        raise ValueError("unsupported or malformed sparse topology schema")\n    entries = blob["entries"]\n    if (not isinstance(entries, dict) or set(entries) != set(modules)\n            or blob["module_count"] != len(modules)):\n        raise ValueError("sparse topology module names/count do not match model")\n    pending = []\n    initialized = 0\n    # Complete validation and device allocation first; malformed later entries\n    # or allocation errors cannot leave earlier modules partially restored.\n    for name, module in modules.items():\n        entry = entries[name]\n        if not isinstance(entry, dict) or set(entry) != {"shape", "packed"}:\n            raise ValueError("malformed sparse topology entry: " + name)\n        shape = _shape(entry["shape"])\n        if shape != tuple(module.weight.shape):\n            raise ValueError("sparse topology mask/weight shape mismatch: " + name)\n        packed = entry["packed"]\n        mask = None if packed is None else unpack_mask(packed, shape, module.weight.device)\n        initialized += mask is not None\n        pending.append((module, mask))\n    if blob["initialized_count"] != initialized:\n        raise ValueError("sparse topology initialized count does not match entries")\n    for module, mask in pending:\n        _install_mask(module, mask)\n    event = {"event": "sparse_topology_resume", "status": "restored_exact",\n             "exact_topology_restored": True, "module_count": len(modules),\n             "initialized_count": initialized,\n             "packed_bytes": sum(0 if e["packed"] is None else e["packed"].numel()\n                                 for e in entries.values())}\n    if emit is not None:\n        emit(event)\n    return event\n')


# --- v6 inproc heldout (triple-goal): measure heldout CE at periodic saves without third CUDA process ---
try:
    import rpv16_inproc_heldout as _rpv16_inproc_heldout
except Exception:
    import importlib.util as _ilu
    from pathlib import Path as _P
    _p=_P(__file__).resolve().parent / "rpv16_inproc_heldout.py"
    if not _p.exists():
        _p=_P("/workspace/claude_opus55_3dff_20260923/inproc_eval/rpv16_inproc_heldout.py")
    _rpv16_inproc_heldout=None
    if _p.exists():
        _spec=_ilu.spec_from_file_location("rpv16_inproc_heldout", str(_p))
        _mod=_ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        _rpv16_inproc_heldout=_mod

def _v6_maybe_inproc_heldout(model, step, ckpt_meta=None):
    """Best-effort; never raises into trainer."""
    if _rpv16_inproc_heldout is None:
        return None
    try:
        if hasattr(_rpv16_inproc_heldout, "run_after_save"):
            return _rpv16_inproc_heldout.run_after_save(model, step, ckpt_meta=ckpt_meta)
        if hasattr(_rpv16_inproc_heldout, "heldout_ce"):
            ce, by_pos, toks = _rpv16_inproc_heldout.heldout_ce(model)
            out={
                "event":"eval","source":"inproc_trainer","step":int(step),
                "ce":{"exact":float(ce)},"tokens":int(toks),
                "by_position":by_pos,"secs":None,
            }
            from pathlib import Path as _P2
            d=_P2("/workspace/rpv16_promote"); d.mkdir(parents=True, exist_ok=True)
            p=d/f"eval_step{int(step)}.json"
            p.write_text(__import__("json").dumps(out))
            return out
    except Exception as e:
        try:
            print(f"[v6-inproc-heldout] failed: {type(e).__name__}: {e}", flush=True)
        except Exception:
            pass
    return None

from flash_attn.ops.triton.layer_norm import rms_norm_fn as _flash_rms_norm_fn

# AGILLM-GB10-1PF v0.1: deliberately hardware-locked for one NVIDIA GB10.
# Conventional parameter count target: ~2B. Training target: 600B tokens.
NAME = "AGILLM-GB10-1PF-targetfix"
# 2026-09-23 cowork: exact downstream credit for rotating stage updates (see Model.forward).
RPV16_E2E_SUFFIX = os.environ.get("AGILLM_RPV16_E2E_SUFFIX", "1") != "0"
# 2026-09-23 cowork: joint update -- every optimizer update trains all 16 stages + token interface + head on the
# exact full-model gradient; stage-wise replay keeps memory at one stage graph (see joint_trunk/joint_backward).
RPV16_JOINT = os.environ.get("AGILLM_RPV16_JOINT", "1") != "0"
JOINT_STAGE = -1
# 2026-09-23 cowork: tied full-vocab classifier (ARCHITECTURE_LOCK token_interface.tied_embedding_classifier=true)
# replaces the rank-16 product-vocab head: logits = (out_proj(norm(x)) @ embed.weight^T) * tied_logit_scale + tied_bias.
RPV16_TIED_HEAD = os.environ.get("AGILLM_RPV16_TIED_HEAD", "1") != "0"
# v86 ALWAYS_HEAD: also train tied classifier (norm/out_proj/bias) on AR rotating-stage updates (no JOINT).
RPV16_ALWAYS_HEAD = os.environ.get("AGILLM_RPV16_ALWAYS_HEAD", "0") != "0"
# v87/v88 SKIP_HEAD_ONLY: with ALWAYS_HEAD, head-only phase is redundant; use all steps for stages.
RPV16_SKIP_HEAD_ONLY = os.environ.get("AGILLM_RPV16_SKIP_HEAD_ONLY", "0") != "0"
# v88b HEAD_LR_MULT: densify tied-head CE under ALWAYS_HEAD without raising trunk LR
RPV16_HEAD_LR_MULT = float(os.environ.get("AGILLM_RPV16_HEAD_LR_MULT", "1.0"))
TIED_CE_CHUNK = int(os.environ.get("AGILLM_TIED_CE_CHUNK", "2048"))
TIED_WARMUP_UPDATES = int(os.environ.get("AGILLM_TIED_WARMUP_UPDATES", "240"))
TIED_WARMUP_LR = float(os.environ.get("AGILLM_TIED_WARMUP_LR", "1e-3"))
TIED_UNIGRAM_PATH = os.environ.get("AGILLM_TIED_UNIGRAM", "/workspace/cowork_loss_diag_20260923/unigram_logp_fwedu.pt")
# Cut-Cross-Entropy (Triton, never materialises [N, V] logits): 0.74-0.89 s fwd+bwd for 49,152 x 129,280 on the shared
# GB10 vs 8.8-9.3 s for the chunked torch path. Unfiltered (filter_eps=None) so every gradient term is kept.
TIED_USE_CCE = os.environ.get("AGILLM_TIED_CCE", "1") != "0"
try:
    from cut_cross_entropy import linear_cross_entropy as _CCE_LCE
except Exception:
    _CCE_LCE = None
# v43d analytical CCE: direct affine bias at native TOKEN_DIM, eliminating exact [1,0x15] padding.
TIED_CCE_AFFINE_BIAS = os.environ.get("AGILLM_TIED_CCE_AFFINE_BIAS", "1") != "0"
try:
    # Direct trainer execution puts this directory on sys.path.
    from tied_cce_bias import linear_cross_entropy_bias as _CCE_BIAS_LCE
except Exception:
    try:
        # Module-style test/import from /workspace/rpv16_sm120.
        from trainer_improve_20261002.tied_cce_bias import linear_cross_entropy_bias as _CCE_BIAS_LCE
    except Exception:
        _CCE_BIAS_LCE = None
TARGET_ALIGNMENT = "next_token_shifted_labels_fullseq_v1"
VOCAB = 129_280
TOKEN_DIM = 256
D = 1_280
STAGES = 16
EXPERTS = 6
FFN = 5_120
Q_HEADS = 20
KV_HEADS = 5
HEAD_DIM = 64
CONTEXT = 2_048
TOTAL_TRAIN_TOKENS = int(os.environ.get("AGILLM_TOTAL_TRAIN_TOKENS", "600000000000"))  # run target from train_env.sh (Scott 2026-10-03: RPV16 = 500B)
VOCAB_GROUPS = 505
VOCAB_LOW = 256
VOCAB_RANK = 16
VOCAB_GROUP_PAD = 512
VOCAB_AFFINE_A = (1,3,7,9,11,13,17,19,21,23,27,29,31,33,37,39)
VOCAB_AFFINE_B = tuple((i * 7919) % VOCAB for i in range(VOCAB_RANK))
SPARSE_PF_TARGET = 1_000_000_000_000_000.0
VERIFIED_SPARSE_TFLOPS = 969.794609
WINDOWS = (256,256,256,256,512,512,512,512,1024,1024,1024,1024,-1,-1,-1,-1)
GB10_MEM_BYTES = 128_427_982_848

ROOT = Path(__file__).resolve().parent
SPARSE_RUNTIME = Path(os.environ.get("AGILLM_GB10_SPARSE_RUNTIME", str(ROOT / "runtime")))
TOKENIZER_JSON = Path(os.environ.get("AGILLM_GB10_TOKENIZER", str(ROOT / "tokenizer_bundle.json")))

FINEWEB_EDU_REVISION = os.environ.get(
    "AGILLM_FINEWEB_EDU_REVISION",
    "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
)

HF_SOURCES = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", 1.60),
    "fineweb": ("HuggingFaceFW/fineweb", "CC-MAIN-2024-10", 0.50),
    "wikipedia": ("wikimedia/wikipedia", "20231101.en", 1.10),
    "c4": ("allenai/c4", "en", 0.40),
    "openwebtext": ("Skylion007/openwebtext", None, 0.35),
    "cosmopedia": ("HuggingFaceTB/cosmopedia", "web_samples_v2", 1.30),
    "smollm-cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 1.30),
    "proof-pile-2": ("EleutherAI/proof-pile-2", "all", 1.50),
    "dolma": ("allenai/dolma", "v1_6-sample", 0.50),
    "codeparrot": ("codeparrot/codeparrot-clean", None, 1.25),
}


def parameter_accounting():
    emb = VOCAB * TOKEN_DIM
    factors = 2 * TOKEN_DIM * D
    expert = STAGES * EXPERTS * 3 * D * FFN
    attn_per = D * D + 2 * D * (KV_HEADS * HEAD_DIM) + D * D
    attn = STAGES * attn_per
    routers = STAGES * D * EXPERTS
    norms = (2 * STAGES + 1) * D
    vocab_factor_head = TOKEN_DIM * (VOCAB_RANK * (VOCAB_GROUP_PAD + VOCAB_LOW) + VOCAB_RANK)
    tied_classifier_extra = VOCAB + 1          # 2026-09-23 cowork: tied_bias + tied_logit_scale (weight = embedding)
    total = emb + factors + expert + attn + routers + norms + vocab_factor_head + tied_classifier_extra
    return {
        "token_embedding": emb,
        "token_factor_projections": factors,
        "experts": expert,
        "attention": attn,
        "routers": routers,
        "norms": norms,
        "rank16_product_vocab_head": vocab_factor_head,
        "tied_classifier_bias_scale": tied_classifier_extra,
        "total": total,
        "target": 2_000_000_000,
        "deviation_percent": 100.0 * (total / 2_000_000_000 - 1.0),
    }


# V91P3_PORTABLE ------------------------------------------------------------------------------------------------
_SPARSE_BACKEND_ENV = os.environ.get("AGILLM_SPARSE_BACKEND", "auto").strip().lower()   # auto | native | emulate
_SPARSE_EMULATE = None


def sparse_backend_emulated():
    """True -> emulated sparse NVFP4 GEMM (any sm_80+ GPU); False -> native Blackwell SM12x kernel."""
    global _SPARSE_EMULATE
    if _SPARSE_EMULATE is None:
        if _SPARSE_BACKEND_ENV == "emulate":
            _SPARSE_EMULATE = True
        elif _SPARSE_BACKEND_ENV == "native":
            _SPARSE_EMULATE = False
        else:
            _SPARSE_EMULATE = torch.cuda.get_device_capability(0)[0] != 12
    return _SPARSE_EMULATE


@triton.jit
def _nvfp4_decode_kernel(P, S, O, K, KB, BLOCK: tl.constexpr):
    # one row x BLOCK columns: E2M1 nibble (low nibble = even column) x fp32 block scale -> BF16 (exact)
    r = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < K
    byte = tl.load(P + r * (K // 2) + cols // 2, mask=m, other=0).to(tl.int32)
    code = tl.where((cols % 2) == 0, byte & 15, (byte >> 4) & 15)
    idx = code & 7
    mag = tl.where(idx < 5, idx.to(tl.float32) * 0.5, tl.where(idx == 5, 3.0, tl.where(idx == 6, 4.0, 6.0)))
    val = mag * tl.where((code & 8) != 0, -1.0, 1.0)   # multiply (not negate): keeps -0.0 for code 0x8 like the reference
    sc = tl.load(S + r * KB + cols // 32, mask=m, other=0.0)
    tl.store(O + r * K + cols, (val * sc).to(tl.bfloat16), mask=m)


_E2M1_LUT_CACHE = {}


def nvfp4_decode_bf16_reference(packed, sf, k):
    """Pure-torch reference of nvfp4_decode_bf16 (used as fallback and by the acceptance test)."""
    key = packed.device
    lut = _E2M1_LUT_CACHE.get(key)
    if lut is None:
        lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                           device=packed.device, dtype=torch.float32)
        _E2M1_LUT_CACHE[key] = lut
    rows = packed.shape[0]
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, k)
    vals = lut[codes.long()].view(rows, k // 32, 32)
    return (vals * sf.float().unsqueeze(-1)).reshape(rows, k).to(torch.bfloat16)


_DECODE_TRITON_OK = None


def nvfp4_decode_bf16(packed, sf, k):
    """NVFP4 codes [rows, k/2] uint8 x E4M3 block-32 scales [rows, k/32] -> BF16 [rows, k] WITHOUT the global
    scale. Exact: an E2M1 value times an E4M3 value has at most 6 significant bits."""
    global _DECODE_TRITON_OK
    if _DECODE_TRITON_OK is not False:
        try:
            rows = packed.shape[0]
            out = torch.empty((rows, k), device=packed.device, dtype=torch.bfloat16)
            BLOCK = 1024
            _nvfp4_decode_kernel[(rows, triton.cdiv(k, BLOCK))](packed.contiguous(), sf.float().contiguous(), out,
                                                                k, k // 32, BLOCK=BLOCK)
            _DECODE_TRITON_OK = True
            return out
        except Exception as e:
            _DECODE_TRITON_OK = False
            print(json.dumps({"event": "nvfp4_decode_triton_fallback", "error": f"{type(e).__name__}: {e}"[:300]}), flush=True)
    return nvfp4_decode_bf16_reference(packed, sf, k)


_MM_F32OUT_OK = None


def _mm_f32acc(a, b):
    """BF16 x BF16 -> FP32 result (tensor cores, FP32 accumulate, no intermediate BF16 rounding)."""
    global _MM_F32OUT_OK
    if _MM_F32OUT_OK is not False:
        try:
            r = torch.mm(a, b, out_dtype=torch.float32)
            _MM_F32OUT_OK = True
            return r
        except (TypeError, RuntimeError, NotImplementedError):
            _MM_F32OUT_OK = False
            print(json.dumps({"event": "emulated_gemm_f32out_unavailable", "note": "falling back to bf16-out mm (one extra BF16 rounding before alpha)"}), flush=True)
    return torch.mm(a, b).float()


@triton.jit
def _emu_scale_cast_kernel(A, ALPHA, O, N, BLOCK: tl.constexpr):
    # V91P4_EMU_EPILOGUE: O = BF16_rne(A * alpha), A fp32 (read once), alpha fp32 scalar on device
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    a = tl.load(A + offs, mask=m, other=0.0)
    al = tl.load(ALPHA)
    tl.store(O + offs, (a * al).to(tl.bfloat16), mask=m)


_EMU_EPILOGUE_OK = None


def _emu_scale_cast(acc, alpha):
    """BF16(acc * alpha) in one pass; bit-identical to acc.mul_(alpha).to(torch.bfloat16)."""
    global _EMU_EPILOGUE_OK
    if _EMU_EPILOGUE_OK is not False and acc.is_contiguous() and alpha.numel() == 1:
        try:
            out = torch.empty(acc.shape, device=acc.device, dtype=torch.bfloat16)
            n = acc.numel()
            BLOCK = 4096
            _emu_scale_cast_kernel[(triton.cdiv(n, BLOCK),)](acc, alpha.reshape(-1)[:1].contiguous(), out, n, BLOCK=BLOCK)
            _EMU_EPILOGUE_OK = True
            return out
        except Exception as e:
            _EMU_EPILOGUE_OK = False
            print(json.dumps({"event": "emu_epilogue_triton_fallback", "error": f"{type(e).__name__}: {e}"[:300]}), flush=True)
    return acc.mul_(alpha).to(torch.bfloat16)

# V91P5_EMU_INT8 ---------------------------------------------------------------------------------------------------
_EMU_GEMM_ENV = os.environ.get("AGILLM_EMU_GEMM", "auto").strip().lower()   # auto | int8 | bf16
_EMU_INT8_SEL = None
_EMU_INT8_OK = None


def emu_gemm_int8():
    global _EMU_INT8_SEL
    if _EMU_INT8_SEL is None:
        if _EMU_GEMM_ENV == "int8":
            _EMU_INT8_SEL = True
        elif _EMU_GEMM_ENV == "bf16":
            _EMU_INT8_SEL = False
        else:
            _EMU_INT8_SEL = ("GeForce" in torch.cuda.get_device_name(0)) and tuple(torch.cuda.get_device_capability(0)) in ((8, 6), (8, 9))
        print(json.dumps({"event": "emu_gemm_select", "int8": _EMU_INT8_SEL, "env": _EMU_GEMM_ENV,
                          "gpu": torch.cuda.get_device_name(0)}), flush=True)
    return _EMU_INT8_SEL


@triton.jit
def _nvfp4_decode_i8_kernel(P, O, K, BLOCK: tl.constexpr):
    # one row x BLOCK columns: E2M1 nibble (low nibble = even column) -> int8 (2 x value), exact
    r = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < K
    byte = tl.load(P + r * (K // 2) + cols // 2, mask=m, other=0).to(tl.int32)
    code = tl.where((cols % 2) == 0, byte & 15, (byte >> 4) & 15)
    i = code & 7
    mag = tl.where(i < 4, i, (2 + (i & 1)) << (((i >> 1) - 1) & 3))
    val = tl.where((code & 8) != 0, -mag, mag)
    tl.store(O + r * K + cols, val.to(tl.int8), mask=m)


def _nvfp4_decode_i8(packed, k):
    rows = packed.shape[0]
    out = torch.empty((rows, k), device=packed.device, dtype=torch.int8)
    _nvfp4_decode_i8_kernel[(rows, triton.cdiv(k, 1024))](packed.contiguous(), out, k, BLOCK=1024)
    return out


def _bs_int8_configs():
    return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "GROUP_M": 8, "U": u}, num_warps=w, num_stages=s)
            for bm, bn, w, s, u in ((128, 128, 8, 3, 1), (128, 128, 8, 4, 1), (128, 256, 8, 3, 1), (256, 128, 8, 3, 1),
                                    (128, 128, 4, 3, 1), (64, 128, 4, 4, 1), (128, 128, 8, 3, 2))]


@triton.jit
def _bs_int8_blk(A, B, SA, SB, a_off, b_off, s_off_a, s_off_b, mm, mn, acc):
    a = tl.load(A + a_off, mask=mm[:, None], other=0)
    b = tl.load(B + b_off, mask=mn[None, :], other=0)
    p = tl.dot(a, b, out_dtype=tl.int32)                                   # exact 32-wide block sum
    sa = tl.load(SA + s_off_a, mask=mm, other=0.0)
    sb = tl.load(SB + s_off_b, mask=mn, other=0.0)
    f = (p + 0x4B400000).to(tl.float32, bitcast=True) - 12582912.0          # exact int32 -> fp32 (|p| < 2^22)
    return acc + f * (sa[:, None] * sb[None, :])                             # exact product, FP32 accumulate


@triton.autotune(configs=_bs_int8_configs(), key=["M", "N", "K"])
@triton.jit
def _bs_int8_gemm_kernel(A, B, SA, SB, ALPHA, C, M, N, K, KB,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr, U: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = rm < M
    mn = rn < N
    rk = tl.arange(0, 32)
    a_off = rm[:, None].to(tl.int64) * K + rk[None, :]
    b_off = rn[None, :].to(tl.int64) * K + rk[:, None]
    sa_off = rm.to(tl.int64) * KB
    sb_off = rn.to(tl.int64) * KB
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, KB, U):
        for u in tl.static_range(U):
            acc = _bs_int8_blk(A, B, SA, SB, a_off + (kb + u) * 32, b_off + (kb + u) * 32, sa_off + kb + u, sb_off + kb + u, mm, mn, acc)
    al = tl.load(ALPHA) * 0.25                                               # (2a)(2w) = 4 a w; /4 is exact
    c_ptrs = C + rm[:, None].to(tl.int64) * N + rn[None, :]
    tl.store(c_ptrs, (acc * al).to(tl.bfloat16), mask=mm[:, None] & mn[None, :])


def _bs_int8_gemm(packed_x, sf_x, packed_w, sf_w, alpha, k):
    xi = _nvfp4_decode_i8(packed_x, k)
    wi = _nvfp4_decode_i8(packed_w, k)
    M, N = xi.shape[0], wi.shape[0]
    out = torch.empty((M, N), device=xi.device, dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _bs_int8_gemm_kernel[grid](xi, wi, sf_x.float().contiguous(), sf_w.float().contiguous(),
                               alpha.reshape(-1)[:1].float().contiguous(), out, M, N, k, k // 32)
    return out
# V91P5_EMU_INT8 end -----------------------------------------------------------------------------------------------


def emulated_sparse_gemm(packed_x, sf_x, packed_w, sf_w, alpha, k):
    """out[n, m] = BF16( alpha * sum_k decode(x)[n,k] * decode(w)[m,k] ), FP32 accumulate: the native SM12x
    block-scaled sparse NVFP4 GEMM's arithmetic (masked weight codes are exact zeros)."""
    global _EMU_INT8_OK
    if _EMU_INT8_OK is not False and k % 128 == 0 and alpha.numel() == 1 and emu_gemm_int8():   # V91P5_EMU_INT8
        try:
            y = _bs_int8_gemm(packed_x, sf_x, packed_w, sf_w, alpha, k)
            _EMU_INT8_OK = True
            return y
        except Exception as e:
            _EMU_INT8_OK = False
            print(json.dumps({"event": "emu_int8_fallback_bf16", "error": f"{type(e).__name__}: {e}"[:300]}), flush=True)
    xd = nvfp4_decode_bf16(packed_x, sf_x, k)
    wd = nvfp4_decode_bf16(packed_w, sf_w, k)
    acc = _mm_f32acc(xd, wd.t())
    del xd, wd
    return _emu_scale_cast(acc, alpha)   # V91P4_EMU_EPILOGUE (was: acc.mul_(alpha).to(torch.bfloat16))


def sparse_backend_info():
    return {"event": "sparse_backend", "backend": "emulated_nvfp4" if sparse_backend_emulated() else "native_sm12x",
            "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "env": _SPARSE_BACKEND_ENV}
# V91P3_PORTABLE end ----------------------------------------------------------------------------------------------


class SparseRuntime:
    def __init__(self, root: Path):
        self.root = root
        # V91P3_PORTABLE: the CUTLASS SM12x GEMM library is only needed by the native backend.
        self.native = None if sparse_backend_emulated() else ctypes.CDLL(str(root / "libsparse_native.so"))
        self.quant = ctypes.CDLL(str(root / "libfused_quant.so"))
        P = ctypes.c_void_p
        n = self.native
        if n is not None:
            n.sg_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]; n.sg_create.restype = P
            n.sg_output_layout.argtypes = []; n.sg_output_layout.restype = ctypes.c_int
            if n.sg_output_layout() != 1:
                raise RuntimeError("sparse runtime requires column-major native output ABI")
            n.sg_pack.argtypes = [P,P,P,P]; n.sg_pack.restype = ctypes.c_int
            n.sg_run_device.argtypes = [P,P,P,P,P,P]; n.sg_run_device.restype = ctypes.c_int
            n.sg_destroy.argtypes = [P]; n.sg_destroy.restype = None
            n.sg_error.argtypes = []; n.sg_error.restype = ctypes.c_char_p
        q = self.quant
        q.sg_global_scale.argtypes = [P,P,P]; q.sg_global_scale.restype = ctypes.c_int
        q.sg_quant32_device_bf16.argtypes = [P,ctypes.c_int,P,P,P,ctypes.c_int64,P,P]
        q.sg_quant32_device_bf16.restype = ctypes.c_int
        q.sg_pair48.argtypes = [P,ctypes.c_int,P,ctypes.c_int64,P]; q.sg_pair48.restype = ctypes.c_int
        q.sg_cuda_error.argtypes = [ctypes.c_int]; q.sg_cuda_error.restype = ctypes.c_char_p

    def check_native(self, rc):
        if rc:
            raise RuntimeError(self.native.sg_error().decode())

    def check_quant(self, rc):
        if rc:
            raise RuntimeError(self.quant.sg_cuda_error(rc).decode())


_RT = None

def rt():
    global _RT
    if _RT is None:
        _RT = SparseRuntime(SPARSE_RUNTIME)
    return _RT


def dtype_kind(x):
    if not x.is_cuda:
        raise ValueError("CUDA tensor required")
    if x.dtype == torch.float32: return 0
    if x.dtype == torch.bfloat16: return 1
    raise ValueError(f"unsupported dtype {x.dtype}")


@torch.no_grad()
def pair48_mask(w):
    w = w.contiguous()
    mask = torch.empty_like(w, dtype=torch.bool)
    rt().check_quant(rt().quant.sg_pair48(w.data_ptr(), dtype_kind(w), mask.data_ptr(), w.numel(), torch.cuda.current_stream().cuda_stream))
    return mask


@torch.no_grad()
def quantize32_bf16(x, materialize=True):
    if x.ndim != 2 or x.shape[-1] % 32 or x.numel() == 0:
        raise ValueError("quantize32 requires nonempty [rows,K], K divisible by 32")
    x = x.contiguous()
    if _AMX_ENABLED:   # V43J AMX: max|x| == max(|min|, max), one reduction, no |x| materialisation
        _mn, _mx = torch.aminmax(x)
        amax = torch.maximum(_mn.abs(), _mx).to(torch.float64)
    else:
        amax = x.abs().amax().to(torch.float64)
    g = torch.empty_like(amax)
    rt().check_quant(rt().quant.sg_global_scale(amax.data_ptr(), g.data_ptr(), torch.cuda.current_stream().cuda_stream))
    packed = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=torch.uint8)
    sf = torch.empty((x.shape[0], x.shape[1] // 32), device=x.device, dtype=torch.float8_e4m3fn)
    dq = torch.empty_like(x, dtype=torch.bfloat16) if materialize else None
    rt().check_quant(rt().quant.sg_quant32_device_bf16(
        x.data_ptr(), dtype_kind(x), packed.data_ptr(), sf.data_ptr(),
        dq.data_ptr() if dq is not None else None, x.numel(), g.data_ptr(),
        torch.cuda.current_stream().cuda_stream))
    return packed, sf, dq, g


_SILU_PACK_LIB = None
def silu_pack_lib():
    global _SILU_PACK_LIB
    if _SILU_PACK_LIB is None:
        lib = ctypes.CDLL('/workspace/rpv16_direct_packed_siluquant_v1/libfused_quant.so')
        P = ctypes.c_void_p
        lib.rpv_silu_mul_amax.argtypes = [P,P,P,ctypes.c_int64,P]; lib.rpv_silu_mul_amax.restype = ctypes.c_int
        lib.rpv_silu_mul_scale.argtypes = [P,P,P]; lib.rpv_silu_mul_scale.restype = ctypes.c_int
        lib.rpv_quant32_silu_mul.argtypes = [P,P,P,P,ctypes.c_int64,P,P]; lib.rpv_quant32_silu_mul.restype = ctypes.c_int
        lib.sg_cuda_error.argtypes = [ctypes.c_int]; lib.sg_cuda_error.restype = ctypes.c_char_p
        _SILU_PACK_LIB = lib
    return _SILU_PACK_LIB


def _silu_pack_check(lib, rc):
    if rc:
        raise RuntimeError(lib.sg_cuda_error(rc).decode())


@torch.no_grad()
def quantize_silu_mul_packed(gate, up):
    if gate.shape != up.shape or gate.ndim != 2 or gate.shape[-1] % 32 or gate.numel() == 0:
        raise ValueError('direct SiLU*up pack requires equal nonempty [rows,K], K divisible by 32')
    if gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16 or not gate.is_cuda or not up.is_cuda:
        raise ValueError('direct SiLU*up pack requires CUDA BF16 inputs')
    # Replacement-host fallback for the old rpv_silu_mul_* fused helper.
    # Preserve the mathematical path exactly: BF16 SiLU(gate)*up followed by
    # the trainer's existing quantize32_bf16 implementation and global scale.
    lib = _sfq_lib() if gate.is_contiguous() and up.is_contiguous() else None   # V43J SFQ
    if lib is not None:
        _V43J_STATS["sfq_calls"] += 1
        return _sfq_run(lib, gate, up)
    _V43J_STATS["sfq_fallbacks"] += 1
    prod = torch.nn.functional.silu(gate.contiguous()) * up.contiguous()
    packed, sf, _dq, g = quantize32_bf16(prod, materialize=False)
    return packed, sf, g


class NativeContext:
    def __init__(self, m, n, k):
        self.m, self.n, self.k = m, n, k
        self.emulated = sparse_backend_emulated()   # V91P3_PORTABLE
        self._pw = self._sfw = None
        self.handle = None
        if self.emulated:
            return
        self.handle = rt().native.sg_create(m, n, k)
        if not self.handle:
            raise RuntimeError(rt().native.sg_error().decode())
    def close(self):
        self._pw = self._sfw = None
        if getattr(self, "handle", None):
            rt().native.sg_destroy(self.handle); self.handle = None
    def __del__(self):
        try: self.close()
        except Exception: pass
    def pack(self, packed_w, sf_w):
        if self.emulated:
            self._pw, self._sfw = packed_w, sf_w
            return
        rt().check_native(rt().native.sg_pack(self.handle, packed_w.data_ptr(), sf_w.data_ptr(), torch.cuda.current_stream().cuda_stream))
    def run(self, packed_x, sf_x, alpha):
        if self.emulated:
            alpha = alpha.to(device=packed_x.device, dtype=torch.float32).contiguous()
            return emulated_sparse_gemm(packed_x, sf_x, self._pw, self._sfw, alpha, self.k)
        out = torch.empty((self.n, self.m), device=packed_x.device, dtype=torch.bfloat16)
        alpha = alpha.to(device=packed_x.device, dtype=torch.float32).contiguous()
        rt().check_native(rt().native.sg_run_device(
            self.handle, packed_x.data_ptr(), sf_x.data_ptr(), out.data_ptr(),
            alpha.data_ptr(), torch.cuda.current_stream().cuda_stream))
        return out


class _SparseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, owner):
        shape = x.shape
        rows = x.numel() // shape[-1]
        k = shape[-1]
        n = (rows + 127) // 128 * 128
        flat = x.reshape(rows, k)
        xp = flat if n == rows else F.pad(flat, (0,0,0,n-rows))
        xpack, xsf, _, xg = quantize32_bf16(xp, materialize=False)
        native, wq, wg = owner.shadow(n)
        y = native.run(xpack, xsf, xg * wg)[:rows]
        ctx.owner = owner
        ctx.shape = shape
        ctx.xdtype = x.dtype
        ctx.save_for_backward(flat)
        return y.reshape(*shape[:-1], owner.out_features).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        o = ctx.owner
        g = gy.reshape(-1, o.out_features).to(torch.bfloat16)
        gx = (g @ o.wq()).reshape(ctx.shape).to(ctx.xdtype) if ctx.needs_input_grad[0] else None
        o.release_wq()
        if ctx.needs_input_grad[1]:
            gw = _cmw_weight_grad(g.t() @ x.to(torch.bfloat16), o)
        else:
            gw = None
        return gx, gw, None


class _SparsePairFn(torch.autograd.Function):
    """Two independent sparse linears sharing one activation quantization.

    Gate and up receive the identical expert input. The production path currently
    quantizes that tensor twice. This function quantizes it once, then executes
    the two ordinary native sparse GEMMs with independent packed weights, weight
    scales, and native contexts. Weight semantics are unchanged.
    """
    @staticmethod
    def forward(ctx, x, w1, w2, owner1, owner2):
        shape = x.shape
        rows = x.numel() // shape[-1]
        k = shape[-1]
        n = (rows + 127) // 128 * 128
        flat = x.reshape(rows, k)
        xp = flat if n == rows else F.pad(flat, (0,0,0,n-rows))
        xpack, xsf, _, xg = quantize32_bf16(xp, materialize=False)
        native1, _, wg1 = owner1.shadow(n)
        native2, _, wg2 = owner2.shadow(n)
        y1 = native1.run(xpack, xsf, xg * wg1)[:rows]
        y2 = native2.run(xpack, xsf, xg * wg2)[:rows]
        ctx.owner1 = owner1
        ctx.owner2 = owner2
        ctx.shape = shape
        ctx.xdtype = x.dtype
        ctx.save_for_backward(flat)
        outshape = (*shape[:-1], owner1.out_features)
        return y1.reshape(outshape).to(x.dtype), y2.reshape(outshape).to(x.dtype)

    @staticmethod
    def backward(ctx, gy1, gy2):
        (x,) = ctx.saved_tensors
        o1, o2 = ctx.owner1, ctx.owner2
        g1 = gy1.reshape(-1, o1.out_features).to(torch.bfloat16)
        g2 = gy2.reshape(-1, o2.out_features).to(torch.bfloat16)
        gx = None
        if ctx.needs_input_grad[0]:
            gx1 = g1 @ o1.wq()
            o1.release_wq()
            gx2 = g2 @ o2.wq()
            o2.release_wq()
            gx = (gx1 + gx2).reshape(ctx.shape).to(ctx.xdtype)
        gw1 = None
        if ctx.needs_input_grad[1]:
            gw1 = _cmw_weight_grad(g1.t() @ x.to(torch.bfloat16), o1)
        gw2 = None
        if ctx.needs_input_grad[2]:
            gw2 = _cmw_weight_grad(g2.t() @ x.to(torch.bfloat16), o2)
        return gx, gw1, gw2, None, None


# ---------------------------------------------------------------------------
# v2b_stagefwd: exact (bitwise) stage-forward restructuring. Each switch set to
# False restores the v1 code path verbatim. main() maps CLI flags onto this.
# ---------------------------------------------------------------------------
STAGEFWD = {
    "syncfree_routing": True,       # --stagefwd-v1-routing restores v1
    "active_fused_silupack": True,  # --stagefwd-v1-active-silu restores v1
    "inplace_repack": True,         # --stagefwd-v1-repack restores v1
}


# ============================================================================ fused SiLU-down backward (claude 2026-09-24)
# Candidate L1: one Triton pass replaces the five ATen elementwise passes of _SiluDownFn.backward
# (silu, act=s*up, gup=ga*s, t=ga*up, silu_backward) -> ~1.1 GB less memory traffic per expert backward, x96 per step.
# Rounding sequence is identical to ATen (fp32 opmath, libdevice expf, IEEE div_rn, one bf16 rounding per stock tensor);
# bitwise-equal witness: /workspace/claude_gb10_improve_20260924/receipts/silu_down_fused_bwd_witness2_20260924T045013Z.json
# Hot toggle: /workspace/claude_gb10_improve_20260924/policy/fused_silu_bwd_policy.json {"enabled": true} read once per
# optimizer update (default OFF -> stock path verbatim). Env AGILLM_FUSED_SILU_BWD=1 forces on, =0 forces off.
_FSB_POLICY_PATH = Path("/workspace/claude_gb10_improve_20260924/policy/fused_silu_bwd_policy.json")
_FSB_ENV = os.environ.get("AGILLM_FUSED_SILU_BWD")
_FSB_ENABLED = False
_FSB_LAST_REPORT = None
_FSB_CALLS = 0
_FSB_FALLBACKS = 0
try:
    import triton as _fsb_triton
    import triton.language as _fsb_tl
    try:
        import triton.language.extra.libdevice as _fsb_ld
    except Exception:
        from triton.language.extra.cuda import libdevice as _fsb_ld
    _FSB_TRITON_OK = True
except Exception:
    _FSB_TRITON_OK = False

if _FSB_TRITON_OK:
    @_fsb_triton.jit
    def _fsb_silu_down_bwd_kernel(gf_ptr, uf_ptr, ga_ptr, act_ptr, gup_ptr, ggate_ptr, n_elements, BLOCK: _fsb_tl.constexpr):
        pid = _fsb_tl.program_id(0)
        offs = pid * BLOCK + _fsb_tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = _fsb_tl.load(gf_ptr + offs, mask=mask, other=0.0).to(_fsb_tl.float32)
        u = _fsb_tl.load(uf_ptr + offs, mask=mask, other=0.0).to(_fsb_tl.float32)
        ga = _fsb_tl.load(ga_ptr + offs, mask=mask, other=0.0).to(_fsb_tl.float32)
        e = _fsb_ld.exp(-x)
        one = 1.0
        denom = one + e
        s = _fsb_ld.div_rn(x, denom).to(_fsb_tl.bfloat16).to(_fsb_tl.float32)      # F.silu(gf) as bf16
        act = (s * u).to(_fsb_tl.bfloat16)                                          # s * uf
        gup = (ga * s).to(_fsb_tl.bfloat16)                                         # ga * s
        t = (ga * u).to(_fsb_tl.bfloat16).to(_fsb_tl.float32)                       # ga * uf (bf16 in stock path)
        sig = _fsb_ld.div_rn(one, denom)                                            # ATen silu_backward sigmoid
        ggate = (t * sig * (one + x * (one - sig))).to(_fsb_tl.bfloat16)
        _fsb_tl.store(act_ptr + offs, act, mask=mask)
        _fsb_tl.store(gup_ptr + offs, gup, mask=mask)
        _fsb_tl.store(ggate_ptr + offs, ggate, mask=mask)


def _fsb_refresh_policy():
    """Read the hot toggle at an update boundary. Never raises; default OFF."""
    global _FSB_ENABLED, _FSB_LAST_REPORT
    enabled = False
    revision = None
    if _FSB_ENV in ("1", "0"):
        enabled = _FSB_ENV == "1"
        revision = "env"
    else:
        try:
            cfg = json.loads(_FSB_POLICY_PATH.read_text())
            enabled = isinstance(cfg, dict) and cfg.get("enabled") is True
            revision = cfg.get("revision") if isinstance(cfg, dict) else None
        except (OSError, ValueError, TypeError):
            enabled = False
    enabled = bool(enabled and _FSB_TRITON_OK)
    _FSB_ENABLED = enabled
    report = (enabled, revision)
    if report != _FSB_LAST_REPORT:
        print(json.dumps({"event": "fused_silu_bwd_config", "enabled": enabled, "triton_ok": _FSB_TRITON_OK,
                          "revision": revision, "ts": time.time()}), flush=True)
        _FSB_LAST_REPORT = report
    return enabled


def _fsb_fused_backward(gf, uf, ga):
    """Returns (act, gup, ggate) bf16 or None if the fused path is not applicable (caller falls back to stock)."""
    global _FSB_CALLS, _FSB_FALLBACKS
    if not (_FSB_ENABLED and _FSB_TRITON_OK and gf.is_cuda and gf.dtype == torch.bfloat16 and uf.dtype == torch.bfloat16
            and ga.dtype == torch.bfloat16 and gf.shape == uf.shape == ga.shape):
        _FSB_FALLBACKS += 1
        return None
    gf = gf.contiguous(); uf = uf.contiguous(); ga = ga.contiguous()
    n = gf.numel()
    act = torch.empty_like(gf); gup = torch.empty_like(gf); ggate = torch.empty_like(gf)
    grid = (_fsb_triton.cdiv(n, 4096),)
    _fsb_silu_down_bwd_kernel[grid](gf, uf, ga, act, gup, ggate, n, BLOCK=4096, num_warps=4)
    _FSB_CALLS += 1
    return act, gup, ggate


def _fsb_telemetry():
    return {"enabled": _FSB_ENABLED, "triton_ok": _FSB_TRITON_OK, "fused_calls": _FSB_CALLS, "stock_calls": _FSB_FALLBACKS}


# Replacement-box compatibility: the original optional mega dgrad+SiLU helper
# was an external hash-pinned speed backend and is not present on this host.
# The trainer already contains an exact stock/fused fallback immediately below.
class _MDSUnavailable:
    @staticmethod
    def enabled():
        return False
    @staticmethod
    def refresh_policy():
        return None
    @staticmethod
    def fused_backward(*args, **kwargs):
        return None
    @staticmethod
    def telemetry():
        return {"enabled": False, "portable_fallback": True, "reason": "external_sg_mds_v38_unavailable"}
_mds_v38 = _MDSUnavailable()


class _GatherRowsFn(torch.autograd.Function):
    """One expert-major row gather feeding all experts.

    `order` is the stable sort permutation of the route, so block i holds exactly the
    rows `(route == i).nonzero()` selected in v1, in the same ascending source
    order. Backward reproduces v1's accumulated index_select gradients: every
    source row receives exactly one expert gradient added onto a zero buffer.
    """
    @staticmethod
    def forward(ctx, flat, order, groups):
        ctx.save_for_backward(order)
        ctx.groups = int(groups)
        ctx.rows = flat.shape[0]
        xs = flat.index_select(0, order)
        return tuple(xs.narrow(0, i * ctx.groups, ctx.groups) for i in range(xs.shape[0] // ctx.groups))

    @staticmethod
    def backward(ctx, *gs):
        (order,) = ctx.saved_tensors
        ref = next(g for g in gs if g is not None)
        gflat = torch.zeros((ctx.rows, ref.shape[-1]), device=ref.device, dtype=ref.dtype)
        for i, g in enumerate(gs):
            if g is not None:
                gflat.index_add_(0, order.narrow(0, i * ctx.groups, ctx.groups), g.reshape(ctx.groups, -1))
        return gflat, None, None


class _ScatterRowsFn(torch.autograd.Function):
    """Scatter the expert outputs back to source-row order (pure copies).

    v1 used six in-place index_copy_ calls whose autograd backward cloned the
    full [rows, D] gradient five times (index_fill) before each index_select.
    The expert row sets are disjoint, so one gather of the incoming gradient is
    value- and bit-identical.
    """
    @staticmethod
    def forward(ctx, order, groups, *ys):
        groups = int(groups)
        ctx.save_for_backward(order)
        ctx.groups = groups
        y = torch.empty((groups * len(ys), ys[0].shape[-1]), device=ys[0].device, dtype=ys[0].dtype)
        for i, yi in enumerate(ys):
            y.index_copy_(0, order.narrow(0, i * groups, groups), yi.reshape(groups, -1))
        return y

    @staticmethod
    def backward(ctx, gy):
        (order,) = ctx.saved_tensors
        gs = gy.index_select(0, order)
        return (None, None) + tuple(gs.narrow(0, i * ctx.groups, ctx.groups) for i in range(gs.shape[0] // ctx.groups))


# Exact BF16-gradient cast + fixed-mask fusion; candidate defaults OFF.
_CMW_POLICY_PATH = Path(os.environ.get("AGILLM_MASKED_WGRAD_POLICY", "/workspace/agillm-gb10-1pf-selftest/policy/masked_wgrad_policy.json"))
_CMW_ENV = os.environ.get("AGILLM_FUSED_MASKED_WGRAD")
_CMW_ENABLED = False
_CMW_LAST_REPORT = None
_CMW_CALLS = 0
_CMW_FALLBACKS = 0
if _FSB_TRITON_OK:
    @_fsb_triton.jit
    def _cmw_cast_mask_kernel(g_ptr, mask_ptr, out_ptr, n, BLOCK: _fsb_tl.constexpr):
        offsets = _fsb_tl.program_id(0) * BLOCK + _fsb_tl.arange(0, BLOCK)
        valid = offsets < n
        g = _fsb_tl.load(g_ptr + offsets, valid, other=0).to(_fsb_tl.float32)
        m = _fsb_tl.load(mask_ptr + offsets, valid, other=0).to(_fsb_tl.float32)
        _fsb_tl.store(out_ptr + offsets, g * m, valid)

def _cmw_refresh_policy():
    global _CMW_ENABLED, _CMW_LAST_REPORT
    enabled, revision = False, None
    if _CMW_ENV in ("0", "1"):
        enabled, revision = _CMW_ENV == "1", "env"
    else:
        try:
            cfg = json.loads(_CMW_POLICY_PATH.read_text())
            if isinstance(cfg, dict):
                enabled, revision = cfg.get("enabled") is True, cfg.get("revision")
        except (OSError, ValueError, TypeError):
            pass
    _CMW_ENABLED = bool(enabled and _FSB_TRITON_OK)
    report = (_CMW_ENABLED, revision)
    if report != _CMW_LAST_REPORT:
        print(json.dumps({"event": "fused_masked_wgrad_config", "enabled": _CMW_ENABLED,
                          "revision": revision, "ts": time.time()}), flush=True)
        _CMW_LAST_REPORT = report
    return _CMW_ENABLED

def _cmw_weight_grad(g, owner):
    global _CMW_CALLS, _CMW_FALLBACKS
    m = owner._fixed_mask
    if (_CMW_ENABLED and _FSB_TRITON_OK and not torch.is_grad_enabled()
            and g.is_cuda and g.dtype == torch.bfloat16 and owner.weight.dtype == torch.float32
            and m is not None and m.dtype == torch.bool and m.device == g.device
            and g.shape == m.shape == owner.weight.shape and g.is_contiguous() and m.is_contiguous()):
        out = torch.empty(g.shape, device=g.device, dtype=torch.float32)
        if g.numel():
            _cmw_cast_mask_kernel[(_fsb_triton.cdiv(g.numel(), 1024),)](g, m, out, g.numel(), BLOCK=1024, num_warps=4)
        _CMW_CALLS += 1
        return out
    _CMW_FALLBACKS += 1
    out = g.to(owner.weight.dtype)
    out.mul_(m)
    return out

def _cmw_telemetry():
    return {"enabled": _CMW_ENABLED, "fused_calls": _CMW_CALLS, "stock_calls": _CMW_FALLBACKS}


# Replacement-box compatibility: optional grouped input pack was an
# execution-only optimization and is default-OFF. Its external .py/.so pair is
# absent here, so preserve the exact ordinary expert path.
_AQ_BACKEND_AVAILABLE = False
def _aq_pack_expert_views(*args, **kwargs):
    raise RuntimeError("grouped-pack backend unavailable on replacement host")
_AQ_ENABLED = False
_AQ_FROZEN_ONLY = True
_AQ_PARTS = 256
_AQ_CALLS = 0
_AQ_POLICY_PATH = Path(os.environ.get(
    'AGILLM_GROUPEDPACK_POLICY',
    '/workspace/agillm-gb10-1pf-selftest/policy/groupedpack_frozen_policy.json'))
_AQ_REVISION = None

def _aq_refresh_policy():
    global _AQ_ENABLED, _AQ_REVISION
    env = os.environ.get('AGILLM_GROUPEDPACK_FROZEN')
    enabled = False
    rev = None
    if env in ('0','1'):
        enabled = env == '1'
        rev = 'env'
    else:
        try:
            cfg = json.loads(_AQ_POLICY_PATH.read_text())
            enabled = isinstance(cfg, dict) and cfg.get('enabled') is True
            rev = cfg.get('revision') if isinstance(cfg, dict) else None
        except (OSError, ValueError, TypeError):
            pass
    _AQ_ENABLED = bool(enabled and _AQ_BACKEND_AVAILABLE)
    _AQ_REVISION = rev
    return _AQ_ENABLED

def _aq_telemetry():
    return {'enabled': bool(_AQ_ENABLED), 'backend_available': bool(_AQ_BACKEND_AVAILABLE), 'revision': _AQ_REVISION,
            'frozen_only': True, 'parts': _AQ_PARTS, 'pack_calls': _AQ_CALLS}

class _GroupedSparsePairFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, owner1, owner2, prepacked):
        shape = x.shape
        rows = x.numel() // shape[-1]
        flat = x.reshape(rows, shape[-1])
        if rows % 128:
            raise ValueError('grouped sparse pair requires complete native row tiles')
        xpack, xsf, xg = prepacked
        native1, _, wg1 = owner1.shadow(rows)
        native2, _, wg2 = owner2.shadow(rows)
        y1 = native1.run(xpack, xsf, xg * wg1)[:rows]
        y2 = native2.run(xpack, xsf, xg * wg2)[:rows]
        ctx.owner1, ctx.owner2 = owner1, owner2
        ctx.shape, ctx.xdtype = shape, x.dtype
        ctx.save_for_backward(flat)
        outshape = (*shape[:-1], owner1.out_features)
        return y1.reshape(outshape).to(x.dtype), y2.reshape(outshape).to(x.dtype)

    @staticmethod
    def backward(ctx, gy1, gy2):
        # Call the exact parent implementation, preserving every GEMM, BF16
        # rounding point, mask operation, and gradient accumulation order.
        return _SparsePairFn.backward(ctx, gy1, gy2) + (None,)


class _SiluDownFn(torch.autograd.Function):
    """Active-stage down projection fed by the fused SiLU*up -> NVFP4 pack.

    Forward emits the identical block32 NVFP4 down-projection input the frozen
    path already uses (no BF16 SiLU*up materialisation, no second read for
    abs/amax/quant). Backward recomputes SiLU(gate)*up with the same ATen
    kernels autograd used in v1 and then evaluates exactly v1's BF16 formulas:
    dW = (g^T @ act) * mask, d_act = g @ Wq, d_up = d_act * silu(gate),
    d_gate = silu_backward(d_act * up, gate).
    """
    @staticmethod
    def forward(ctx, gate, up, w, owner):
        shape = gate.shape
        gf = gate.reshape(-1, owner.in_features)
        uf = up.reshape(-1, owner.in_features)
        rows = gf.shape[0]
        packed, sf, xg = quantize_silu_mul_packed(gf, uf)
        native, _, wg = owner.shadow(rows)
        y = native.run(packed, sf, xg * wg)[:rows]
        ctx.owner = owner
        ctx.shape = shape
        ctx.xdtype = gate.dtype
        ctx.save_for_backward(gf, uf)
        return y.reshape(*shape[:-1], owner.out_features).to(gate.dtype)

    @staticmethod
    def backward(ctx, gy):
        gf, uf = ctx.saved_tensors
        o = ctx.owner
        g = gy.reshape(-1, o.out_features).to(torch.bfloat16)
        if _mds_v38.enabled() and ctx.needs_input_grad[2] and ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            mega = _mds_v38.fused_backward(g, o.wq(), gf, uf)
            o.release_wq()
            if mega is not None:
                act, gup_f, ggate_f = mega
                gw = _cmw_weight_grad(g.t() @ act, o)
                del act
                return ggate_f.reshape(ctx.shape), gup_f.reshape(ctx.shape), gw, None
        if _FSB_ENABLED and ctx.needs_input_grad[2] and (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            # Exact fallback retained verbatim when mega-kernel is disabled/ineligible.
            ga = (g @ o.wq()).reshape(gf.shape).to(ctx.xdtype)
            o.release_wq()
            fused = _fsb_fused_backward(gf, uf, ga)
            if fused is not None:
                act, gup_f, ggate_f = fused
                gw = _cmw_weight_grad(g.t() @ act, o)
                del act
                gup = gup_f.reshape(ctx.shape) if ctx.needs_input_grad[1] else None
                ggate = ggate_f.reshape(ctx.shape) if ctx.needs_input_grad[0] else None
                return ggate, gup, gw, None
        s = F.silu(gf)
        gw = None
        if ctx.needs_input_grad[2]:
            act = s * uf
            gw = _cmw_weight_grad(g.t() @ act.to(torch.bfloat16), o)
            del act
        ggate = gup = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ga = (g @ o.wq()).reshape(gf.shape).to(ctx.xdtype)
            o.release_wq()
            if ctx.needs_input_grad[1]:
                gup = (ga * s).reshape(ctx.shape)
            if ctx.needs_input_grad[0]:
                ggate = torch.ops.aten.silu_backward(ga * uf, gf).reshape(ctx.shape)
        return ggate, gup, gw, None



# ============================================================================ cowork-fable v43j (2026-10-04)
# Execution-only speedups for the RTX 3090 portable path (Scott: "apply it then, what you think is best").
# No model, objective, optimizer, data, schedule or checkpoint-schema change. Every switch defaults ON, can be
# turned off by env, and falls back to the exact v43i code path whenever its preconditions do not hold:
#  FSX  frozen E2E-suffix stage as ONE autograd node. Forward = the stage's own no-grad forward (identical kernels
#       and values to the checkpoint first pass). Backward recomputes only what the frozen-weight VJP needs
#       (n1 -> attention -> n2 graph, routing order, per-expert gate/up GEMMs one expert at a time) and applies the
#       v1 formulas by hand: scatter VJP = gather(gy), down dgrad = g @ Wq, _SiluDownFn stock SiLU VJP (dgrad-only
#       Triton kernel with the witnessed ATen rounding), _SparsePairFn dgrad (g1@Wq1 + g2@Wq2), _GatherRowsFn
#       index_add. Skips per suffix stage: 6 down GEMMs, the SiLU*up NVFP4 pack, the router softmax/aux graph and
#       the scatter, and holds one expert's gate/up at a time instead of six (lower peak memory).
#  WQD  SparseLinear.wq() decodes the cached NVFP4 pack with the quantizer's own dq formula
#       bf16_rn(level * RN(e4m3(sf) * float(g))) instead of re-quantizing weight*mask on every backward.
#  AMX  quantize32_bf16 global amax via one aminmax reduction (max(|min|, max), exact) instead of |x| + amax.
#  SFQ  quantize_silu_mul_packed via an sm_86 build of rpv16_siluquant_sm120.cu (fused SiLU*up amax + pack, the
#       documented bit-identical contract); self-tested bitwise against the unfused path on first use.
_FSX_ENABLED = os.environ.get("AGILLM_FROZEN_SUFFIX_FN", "1") != "0"
_WQD_ENABLED = os.environ.get("AGILLM_WQ_DECODE", "1") != "0"
_AMX_ENABLED = os.environ.get("AGILLM_AMAX_FUSED", "1") != "0"
_SFQ_ENABLED = os.environ.get("AGILLM_SILU_QUANT_LIB", "1") != "0"
_SFQ_PATH = os.environ.get("AGILLM_SILU_QUANT_LIB_PATH",
                           "/workspace/cowork_fable_speedup_20261004/lib/libsiluquant_sm86.so")
_V43J_STATS = {"fsx_calls": 0, "fsx_fallbacks": 0, "wqd_calls": 0, "wqd_fallbacks": 0,
               "sfq_calls": 0, "sfq_fallbacks": 0, "sfq_selftest": None}
try:
    import triton.language.extra.libdevice as _v43j_ld
except Exception:
    from triton.language.extra.cuda import libdevice as _v43j_ld


@triton.jit
def _wqd_kernel(P, S, G, O, K, BLOCK: tl.constexpr):
    # One row x BLOCK columns of an NVFP4 pack -> the exact BF16 dq quant32_kernel_bf16dq writes:
    #   dq = bf16_rn((negative ? -level : level) * (float(e4m3 sf) * float(g)))
    r = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < K
    byte = tl.load(P + r * (K // 2) + cols // 2, mask=m, other=0).to(tl.int32)
    code = tl.where((cols % 2) == 0, byte & 15, (byte >> 4) & 15)
    idx = code & 7
    lvl = tl.where(idx < 5, idx.to(tl.float32) * 0.5, tl.where(idx == 5, 3.0, tl.where(idx == 6, 4.0, 6.0)))
    lvl = lvl * tl.where((code & 8) != 0, -1.0, 1.0)              # multiply, not negate: keeps -0.0 like the quantizer
    sb = tl.load(S + r * (K // 32) + cols // 32, mask=m, other=0).to(tl.int32)
    e = (sb >> 3) & 15
    mt = (sb & 7).to(tl.float32)
    p2 = ((e + 117) << 23).to(tl.float32, bitcast=True)            # 2^(e-10), exact (e >= 1)
    sfv = tl.where(e == 0, mt * 0.001953125, (8.0 + mt) * p2)      # e4m3 subnormal m*2^-9 / normal (8+m)*2^(e-10)
    sfv = tl.where((sb & 127) == 127, float("nan"), sfv)
    g = tl.load(G).to(tl.float32)
    unit = sfv * g
    tl.store(O + r * K + cols, (lvl * unit).to(tl.bfloat16), mask=m)


def _wqd_decode(packed, sf, g, k):
    rows = packed.shape[0]
    out = torch.empty((rows, k), device=packed.device, dtype=torch.bfloat16)
    _wqd_kernel[(rows, triton.cdiv(k, 1024))](packed.contiguous(), sf.contiguous().view(torch.uint8),
                                              g.reshape(-1)[:1].contiguous(), out, k, BLOCK=1024)
    return out


@triton.jit
def _fsx_silu_dgrad_kernel(gf_ptr, uf_ptr, ga_ptr, gup_ptr, ggate_ptr, n_elements, BLOCK: tl.constexpr):
    # _fsb_silu_down_bwd_kernel minus the wgrad-only `act` output (same ops, same rounding points).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(gf_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(uf_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ga = tl.load(ga_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    e = _v43j_ld.exp(-x)
    one = 1.0
    denom = one + e
    s = _v43j_ld.div_rn(x, denom).to(tl.bfloat16).to(tl.float32)
    gup = (ga * s).to(tl.bfloat16)
    t = (ga * u).to(tl.bfloat16).to(tl.float32)
    sig = _v43j_ld.div_rn(one, denom)
    ggate = (t * sig * (one + x * (one - sig))).to(tl.bfloat16)
    tl.store(gup_ptr + offs, gup, mask=mask)
    tl.store(ggate_ptr + offs, ggate, mask=mask)


_FSX_DGRAD_FUSED = os.environ.get("AGILLM_FSX_FUSED_DGRAD", "1") != "0"


def _fsx_silu_dgrad(gf, uf, ga):
    """Frozen-weight _SiluDownFn VJP: (gup, ggate) = (ga*silu(gf), silu_backward(ga*uf, gf)), bf16."""
    if _FSX_DGRAD_FUSED and gf.is_contiguous() and uf.is_contiguous() and ga.is_contiguous():
        n = gf.numel()
        gup = torch.empty_like(gf); ggate = torch.empty_like(gf)
        _fsx_silu_dgrad_kernel[(triton.cdiv(n, 4096),)](gf, uf, ga, gup, ggate, n, BLOCK=4096, num_warps=4)
        return gup, ggate
    s = F.silu(gf)
    return ga * s, torch.ops.aten.silu_backward(ga * uf, gf)


def _fsx_route_order(stage, flat):
    # Stage.forward's balanced affine routing, verbatim (deterministic: same input -> same order).
    logits = stage.router(flat).float()
    groups = flat.shape[0] // EXPERTS
    gl = logits.view(EXPERTS, groups, EXPERTS).permute(1, 0, 2).contiguous()
    perms = stage._route_perms
    pick = perms.view(1, perms.shape[0], EXPERTS, 1).expand(groups, -1, -1, 1)
    scores = gl.unsqueeze(1).expand(-1, perms.shape[0], -1, -1).gather(3, pick).squeeze(3).sum(2)
    best = scores.argmax(1)
    route = perms.index_select(0, best).transpose(0, 1).reshape(-1)
    return torch.sort(route.to(torch.uint8), stable=True)[1], groups


class _FrozenStageFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, stage):
        with torch.no_grad():
            y, _aux, _counts = stage(x)
        ctx.stage = stage
        ctx.attn = (_AH_ATTN.mode, _AH_ATTN.first2)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        st = ctx.stage
        prev = (_AH_ATTN.mode, _AH_ATTN.first2)
        _AH_ATTN.mode, _AH_ATTN.first2 = ctx.attn
        try:
            with torch.enable_grad():
                xi = x.detach().requires_grad_(True)
                xm = xi + st.attn(st.n1(xi))
                h = st.n2(xm)
            with torch.no_grad():
                flat = h.detach().reshape(-1, D)
                order, groups = _fsx_route_order(st, flat)
                xs_all = flat.index_select(0, order)                      # _GatherRowsFn.forward
                gys = gy.reshape(-1, D).index_select(0, order)            # residual-scatter VJP (expert part)
                dflat = torch.zeros((flat.shape[0], D), device=flat.device, dtype=flat.dtype)
                for i, e in enumerate(st.experts):
                    xe = xs_all.narrow(0, i * groups, groups)
                    gate, up = _SparsePairFn.apply(xe, e.gate.weight, e.up.weight, e.gate, e.up)
                    o = e.down
                    g = gys.narrow(0, i * groups, groups).reshape(-1, o.out_features).to(torch.bfloat16)
                    gf = gate.reshape(-1, o.in_features)
                    uf = up.reshape(-1, o.in_features)
                    ga = (g @ o.wq()).reshape(gf.shape).to(gate.dtype)
                    o.release_wq()
                    gup, ggate = _fsx_silu_dgrad(gf, uf, ga)
                    del gate, up, gf, uf, ga, g
                    g1 = ggate.reshape(-1, e.gate.out_features).to(torch.bfloat16)
                    g2 = gup.reshape(-1, e.up.out_features).to(torch.bfloat16)
                    gx1 = g1 @ e.gate.wq()
                    e.gate.release_wq()
                    gx2 = g2 @ e.up.wq()
                    e.up.release_wq()
                    gx = (gx1 + gx2).reshape(xe.shape).to(xe.dtype)
                    del g1, g2, gx1, gx2, gup, ggate
                    dflat.index_add_(0, order.narrow(0, i * groups, groups), gx.reshape(groups, -1))
                    del gx
                del xs_all, gys
            torch.autograd.backward([xm, h], [gy, dflat.view_as(h)])
            dx = xi.grad
        finally:
            _AH_ATTN.mode, _AH_ATTN.first2 = prev
        _V43J_STATS["fsx_calls"] += 1
        return dx, None


def _fsx_eligible(stage, x):
    if not (_FSX_ENABLED and torch.is_grad_enabled() and x.requires_grad and x.is_cuda):
        return False
    if _AH_ATTN.mode != "ar" or _AQ_ENABLED:
        return False
    if not (STAGEFWD["syncfree_routing"] and STAGEFWD["active_fused_silupack"]):
        return False
    rows = x.numel() // D
    if x.shape[-1] != D or rows % EXPERTS or (rows // EXPERTS) % 128:
        return False
    return not any(p.requires_grad for p in stage.parameters())


def _frozen_suffix_stage(stage, x):
    """One frozen E2E-suffix stage: same output, same input gradient as checkpoint(stage, x)[0]."""
    if _fsx_eligible(stage, x):
        y = _FrozenStageFn.apply(x, stage)
        return y, [x.numel() // D // EXPERTS] * EXPERTS
    _V43J_STATS["fsx_fallbacks"] += 1
    from torch.utils.checkpoint import checkpoint as _e2e_ckpt
    y, _aux, counts = _e2e_ckpt(stage, x, use_reentrant=False)
    return y, counts


_SFQ_LIB = None
_SFQ_STATE = "untested"


def _sfq_lib():
    """sm_86 fused SiLU*up -> NVFP4 pack library, bitwise self-tested once against the unfused v43i path."""
    global _SFQ_LIB, _SFQ_STATE
    if not _SFQ_ENABLED or _SFQ_STATE == "failed":
        return None
    if _SFQ_LIB is not None:
        return _SFQ_LIB
    try:
        lib = ctypes.CDLL(_SFQ_PATH)
        P = ctypes.c_void_p
        lib.rpv_silu_mul_amax.argtypes = [P, P, P, ctypes.c_int64, P]; lib.rpv_silu_mul_amax.restype = ctypes.c_int
        lib.rpv_silu_mul_scale.argtypes = [P, P, P]; lib.rpv_silu_mul_scale.restype = ctypes.c_int
        lib.rpv_quant32_silu_mul.argtypes = [P, P, P, P, ctypes.c_int64, P, P]
        lib.rpv_quant32_silu_mul.restype = ctypes.c_int
        lib.sg_cuda_error.argtypes = [ctypes.c_int]; lib.sg_cuda_error.restype = ctypes.c_char_p
        gen = torch.Generator(device="cuda").manual_seed(4321)
        ok = True
        for rows, scale in ((256, 1.0), (512, 8.0), (128, 0.02)):
            gt = (torch.randn((rows, FFN), device="cuda", generator=gen) * scale).to(torch.bfloat16)
            ut = (torch.randn((rows, FFN), device="cuda", generator=gen) * scale).to(torch.bfloat16)
            gt.view(-1)[:7] = torch.tensor([0.0, -0.0, 1e-30, -88.0, 88.0, 3.0, -3.0], device="cuda").to(torch.bfloat16)
            p0, s0, g0 = _sfq_run(lib, gt, ut)
            prod = torch.nn.functional.silu(gt) * ut
            p1, s1, _dq, g1 = quantize32_bf16(prod, materialize=False)
            ok = ok and torch.equal(p0, p1) and torch.equal(s0.view(torch.uint8), s1.view(torch.uint8)) \
                and torch.equal(g0, g1)
        _SFQ_STATE = "passed" if ok else "failed"
        _V43J_STATS["sfq_selftest"] = _SFQ_STATE
        print(json.dumps({"event": "v43j_silu_quant_lib", "path": _SFQ_PATH, "selftest": _SFQ_STATE}), flush=True)
        if ok:
            _SFQ_LIB = lib
            return lib
    except Exception as exc:
        _SFQ_STATE = "failed"
        _V43J_STATS["sfq_selftest"] = "error: " + f"{type(exc).__name__}: {exc}"[:160]
        print(json.dumps({"event": "v43j_silu_quant_lib", "path": _SFQ_PATH, "selftest": "error",
                          "error": f"{type(exc).__name__}: {exc}"[:200]}), flush=True)
    return None


def _sfq_run(lib, gate, up):
    gate = gate.contiguous(); up = up.contiguous()
    rows, k = gate.shape
    n = gate.numel()
    amax = torch.empty((), device=gate.device, dtype=torch.float32)
    g = torch.empty((), device=gate.device, dtype=torch.float64)
    packed = torch.empty((rows, k // 2), device=gate.device, dtype=torch.uint8)
    sf = torch.empty((rows, k // 32), device=gate.device, dtype=torch.float8_e4m3fn)
    stream = torch.cuda.current_stream().cuda_stream

    def chk(rc):
        if rc:
            raise RuntimeError(lib.sg_cuda_error(rc).decode())
    chk(lib.rpv_silu_mul_amax(gate.data_ptr(), up.data_ptr(), amax.data_ptr(), n, stream))
    chk(lib.rpv_silu_mul_scale(amax.data_ptr(), g.data_ptr(), stream))
    chk(lib.rpv_quant32_silu_mul(gate.data_ptr(), up.data_ptr(), packed.data_ptr(), sf.data_ptr(), n,
                                 g.data_ptr(), stream))
    return packed, sf, g


def v43j_telemetry():
    return dict(_V43J_STATS, fsx=_FSX_ENABLED, wqd=_WQD_ENABLED, amx=_AMX_ENABLED, sfq=_SFQ_ENABLED,
                sfq_state=_SFQ_STATE)


print(json.dumps({"event": "v43j_execution_speedups", "schema": "cowork-fable.v43j.v1", "fsx": _FSX_ENABLED,
                  "wqd": _WQD_ENABLED, "amx": _AMX_ENABLED, "sfq": _SFQ_ENABLED, "sfq_path": _SFQ_PATH}), flush=True)
# ============================================================================ end cowork-fable v43j block


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        if in_features % 256 or out_features % 128:
            raise ValueError(f"sparse shape must satisfy K%256=0 and M%128=0, got {out_features}x{in_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device="cuda", dtype=torch.float32))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self._fixed_mask = None
        self._cache_key = None
        self._wq = None
        self._wg = None
        self._packed_w = None
        self._sf_w = None
        self._contexts = OrderedDict()
        self.pack_count = 0
        self.kernel_calls = 0

    @torch.no_grad()
    def wq(self):
        # V91P2: BF16 dequantised masked weight, built on demand for the backward GEMM only.
        # Identical to what shadow() used to keep resident: same weight version, same fixed mask,
        # same deterministic quantize32_bf16(..., materialize=True) kernel.
        key = (self.weight.data_ptr(), self.weight._version)
        if self._wq is None or getattr(self, '_wq_key', None) != key:
            if self._fixed_mask is None:
                self._fixed_mask = pair48_mask(self.weight)
            if (_WQD_ENABLED and self._cache_key == key and self._packed_w is not None
                    and self._sf_w is not None and self._wg is not None):
                self._wq = _wqd_decode(self._packed_w, self._sf_w, self._wg, self.in_features)   # V43J WQD
                _V43J_STATS["wqd_calls"] += 1
            else:
                _p, _s, self._wq, _g = quantize32_bf16(self.weight * self._fixed_mask, materialize=True)
                _V43J_STATS["wqd_fallbacks"] += 1
            self._wq_key = key
        return self._wq

    def release_wq(self):
        self._wq = None

    @torch.no_grad()
    def shadow(self, n):
        if self._fixed_mask is None:
            self._fixed_mask = pair48_mask(self.weight)
        key = (self.weight.data_ptr(), self.weight._version)
        if self._cache_key != key:
            masked = self.weight * self._fixed_mask
            self._packed_w, self._sf_w, _unused_dq, self._wg = quantize32_bf16(masked, materialize=False)  # V91P2: _wq lazy
            self._wq = None
            self._cache_key = key
            if STAGEFWD["inplace_repack"]:
                # The native State::pack is re-callable: it rewrites the packed
                # weight, the sparsity metadata and the weight block scales in
                # the buffers sg_create allocated. Re-packing in place replaces
                # v1's sg_destroy (6 cudaFree = 6 device syncs) + sg_create
                # (4 cudaMalloc + 2 cudaMemset) + 2 lazy workspace cudaMalloc
                # per layer per weight version. Shapes never change, so the
                # buffers, layouts and workspaces are identical.
                for c in self._contexts.values():
                    c.pack(self._packed_w, self._sf_w)
                    self.pack_count += 1
            else:
                for c in self._contexts.values(): c.close()
                self._contexts.clear()
        c = self._contexts.get(n)
        if c is None:
            # One native context per sparse layer: the production router uses a
            # fixed 8192-row expert shape, and this hard cap prevents accidental
            # shape drift from leaking GBs of native workspace.
            for old_ctx in self._contexts.values():
                old_ctx.close()
            self._contexts.clear()
            c = NativeContext(self.out_features, n, self.in_features)
            c.pack(self._packed_w, self._sf_w)
            self._contexts[n] = c
            self.pack_count += 1
        self.kernel_calls += 1
        return c, self._wq, self._wg

    def forward(self, x):
        return _SparseFn.apply(x, self.weight, self)

    @torch.no_grad()
    def forward_packed(self, packed_x, sf_x, xg, rows):
        n = (int(rows) + 127) // 128 * 128
        if n != int(rows):
            raise ValueError('direct packed sparse path currently requires rows divisible by 128')
        native, _, wg = self.shadow(n)
        return native.run(packed_x, sf_x, xg * wg)[:int(rows)]

    def invalidate(self):
        self._cache_key = None


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d, device="cuda", dtype=torch.bfloat16))
    def forward(self, x):
        return _flash_rms_norm_fn(x, self.weight, None, eps=1e-5)


def alibi_slopes(n):
    def slopes_pow2(k):
        start = 2 ** (-2 ** -(math.log2(k) - 3))
        ratio = start
        return [start * ratio ** i for i in range(k)]
    if math.log2(n).is_integer(): vals = slopes_pow2(n)
    else:
        p = 2 ** math.floor(math.log2(n))
        vals = slopes_pow2(p) + slopes_pow2(2*p)[0::2][:n-p]
    return torch.tensor(vals, device="cuda", dtype=torch.float32)


class Attention(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.q = nn.Linear(D, D, bias=False, device="cuda", dtype=torch.bfloat16)
        self.k = nn.Linear(D, KV_HEADS*HEAD_DIM, bias=False, device="cuda", dtype=torch.bfloat16)
        self.v = nn.Linear(D, KV_HEADS*HEAD_DIM, bias=False, device="cuda", dtype=torch.bfloat16)
        self.o = nn.Linear(D, D, bias=False, device="cuda", dtype=torch.bfloat16)
    def forward(self, x):
        from flash_attn import flash_attn_func
        b,s,_ = x.shape
        q = self.q(x).view(b,s,Q_HEADS,HEAD_DIM)
        k = self.k(x).view(b,s,KV_HEADS,HEAD_DIM)
        v = self.v(x).view(b,s,KV_HEADS,HEAD_DIM)
        if _AH_ATTN.mode == "ar":
            ws = (-1,-1) if self.window < 0 else (self.window, 0)
            y = flash_attn_func(q,k,v,dropout_p=0.0,causal=True,window_size=ws)
        else:
            y = ah_attention(q,k,v,self.window)
        return self.o(y.reshape(b,s,D))


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = SparseLinear(D, FFN)
        self.up = SparseLinear(D, FFN)
        self.down = SparseLinear(FFN, D)
    def forward(self, x, _prepacked=None):
        if _prepacked is None:
            gate, up = _SparsePairFn.apply(x, self.gate.weight, self.up.weight, self.gate, self.up)
        else:
            gate, up = _GroupedSparsePairFn.apply(x, self.gate.weight, self.up.weight, self.gate, self.up, _prepacked)
        # Frozen prefix/head-anchor stages execute under no_grad. Emit the exact
        # block32 NVFP4 down-projection input directly, avoiding the 8192x5120
        # BF16 SiLU*up materialization and the subsequent reread/requantize.
        # The active trainable stage keeps the original autograd path unchanged.
        if not gate.requires_grad and not up.requires_grad:
            gf = gate.reshape(-1, FFN)
            uf = up.reshape(-1, FFN)
            packed, sf, xg = quantize_silu_mul_packed(gf, uf)
            y = self.down.forward_packed(packed, sf, xg, gf.shape[0])
            return y.reshape(*x.shape[:-1], D).to(x.dtype)
        if (STAGEFWD["active_fused_silupack"] and gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16
                and gate.numel() and (gate.numel() // FFN) % 128 == 0):
            return _SiluDownFn.apply(gate, up, self.down.weight, self.down)
        return self.down(F.silu(gate) * up)


class Stage(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.n1 = RMSNorm(D); self.n2 = RMSNorm(D)
        self.attn = Attention(window)
        self.router = nn.Linear(D, EXPERTS, bias=False, device="cuda", dtype=torch.bfloat16)
        self.experts = nn.ModuleList([Expert() for _ in range(EXPERTS)])
        # 12 affine permutations of six experts. Every candidate assigns one
        # token in each six-token hardware group to every expert, guaranteeing
        # exact 8192-row expert GEMMs for the 24x2048 production geometry.
        perms = [[(a*j + b) % EXPERTS for j in range(EXPERTS)]
                 for a in (1, EXPERTS - 1) for b in range(EXPERTS)]
        self.register_buffer("_route_perms",
            torch.tensor(perms, device="cuda", dtype=torch.long), persistent=False)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        h = self.n2(x)
        flat = h.reshape(-1, D)
        logits = self.router(flat).float()
        probs = logits.softmax(-1)
        if flat.shape[0] % EXPERTS == 0:
            groups = flat.shape[0] // EXPERTS
            # Group tokens spaced evenly through the flattened batch, then pick
            # the highest-scoring balanced affine assignment per group. This is
            # exact-capacity top-1 routing with zero expert-shape variance.
            gl = logits.view(EXPERTS, groups, EXPERTS).permute(1, 0, 2).contiguous()
            perms = self._route_perms
            pick = perms.view(1, perms.shape[0], EXPERTS, 1).expand(groups, -1, -1, 1)
            scores = gl.unsqueeze(1).expand(-1, perms.shape[0], -1, -1).gather(3, pick).squeeze(3).sum(2)
            best = scores.argmax(1)
            route = perms.index_select(0, best).transpose(0, 1).reshape(-1)
        else:
            route = probs.argmax(-1)
        residual_included = STAGEFWD["syncfree_routing"] and flat.shape[0] % EXPERTS == 0
        if residual_included:
            # Balanced affine routing assigns every expert exactly one row of
            # every six-row group, so each expert owns exactly `groups` rows by
            # construction and no host-side count is needed. A stable sort of
            # the route is expert-major and preserves ascending source order
            # inside each expert block, i.e. block i == (route == i).nonzero()
            # of v1 element for element: expert GEMM inputs and the BF16 dW
            # reduction order are unchanged. Zero host syncs. (uint8 keys: one
            # radix pass; measured ~3x cheaper than the int64 argsort and ~5x
            # cheaper than v1's six nonzero scans, identical permutation.)
            order = torch.sort(route.to(torch.uint8), stable=True)[1]
            xs = _GatherRowsFn.apply(flat, order, groups)
            if _AQ_ENABLED and (not _AQ_FROZEN_ONLY or not torch.is_grad_enabled()) and groups % 128 == 0:
                global _AQ_CALLS
                _qpacks = _aq_pack_expert_views(xs, parts=_AQ_PARTS)
                _AQ_CALLS += 1
                ys = [e(xi, qp) for e, xi, qp in zip(self.experts, xs, _qpacks)]
            else:
                ys = [e(xi) for e, xi in zip(self.experts, xs)]
            from sg_moe_residual_d803c31d import scatter_add
            y = scatter_add(x, order, groups, ys, _ScatterRowsFn.apply, emit=_emit)
            counts = [groups] * EXPERTS
        else:
            y = torch.empty_like(flat)
            counts = []
            for i,e in enumerate(self.experts):
                idx = (route == i).nonzero(as_tuple=False).flatten()
                counts.append(int(idx.numel()))
                if idx.numel():
                    yi = e(flat.index_select(0, idx))
                    y.index_copy_(0, idx, yi)
        # Routing is exactly balanced by construction, so train the router by
        # reinforcing the selected balanced assignment rather than a balance loss
        # that would be constant under exact capacities.
        selected = probs.gather(1, route[:, None]).squeeze(1).clamp_min(1e-9)
        aux = -selected.log().mean()
        return (y if residual_included else x + y.view_as(x)), aux, counts




# >>> FUSED_CE_EMBED BEGIN (source: fused_ce/rpv16_fused_ce.py)
# rpv16_fused_ce.py -- chunked FUSED LINEAR + rank-16 product-vocab CE (Liger-style) for AGILLM-GB10-1PF.
#
# Drop-in replacement for the inline RPV16 CE block of the production trainer
# (agillm_gb10_1pf.rpv16_affine_sharedquant_phaseclock_datacursor_targetfix_prod_v1.py, Model.forward,
# "if labels is not None:" branch).  Production semantics reproduced exactly (up to float rounding):
#
#   hh = h.reshape(-1, 256) (bf16), yy = labels.reshape(-1) (int64, every row counts, NO ignore mask)
#   per chunk of ce_chunk rows:  raw = F.linear(z, W_factor)  [bf16 GEMM, no bias]  -> FP32
#       group logits raw[:, r, :512] with classes 505..511 forced to -1e9, low logits raw[:, r, 512:768]
#       code_r = (token*a_r + b_r) mod V ; group_r = code_r // 256 ; low_r = code_r % 256
#       lm = log_softmax(F.linear(z, W_mix).float())
#       nll_row = -logsumexp_r(lm_r + log_softmax(group_r)[group_r] + log_softmax(low_r)[low_r])
#   ce = (sum over chunks of chunk-sums) / yy.numel()            (FP32 scalar)
#
# Integration (later step, NOT done here) is one line in Model.forward:
#   ce = rpv16_fused_ce(h.reshape(-1, TOKEN_DIM), labels.reshape(-1),
#                       self.factor_head.weight, self.mix_head.weight,
#                       ce_chunk=int(getattr(self, "ce_chunk", 4096)))      # (+ grad_scale_hint=1/grad_accum if != 1)
#   loss = ce + 0.01 * aux_total / STAGES
#
# Implementation: per chunk one BF16 GEMM for the 12288 factor logits + one for the 16 mix logits, then ONE
# Triton kernel (one program per row) that derives the 16 affine targets with integer math, does the masked
# group log-softmax (505 of 512), the low log-softmax (256), the target gathers and the rank logsumexp with
# FP32 accumulators, writes the per-row NLL and overwrites the logits / mix-logits buffers IN PLACE with
# d(ce)/dlogits (already scaled by 1/N in FP32 *before* the cast to the logits dtype, exactly where
# production's autograd does its FP32->BF16 cast).  dW / dh are accumulated with GEMMs of the same dtype and
# the same shapes production's autograd uses.  No FP32 [chunk,16,768] tensor, no checkpoint recompute, no
# slice copies.  backward() only scales the stored grads by grad_output.
#
# Gradients are only computed for inputs that require grad (stage steps: hidden rows only; head-anchor
# step: hidden rows + factor_head.weight + mix_head.weight; no_grad/eval: loss only).

import os

import torch
import triton
import triton.language as tl

VOCAB = 129_280
VOCAB_GROUPS = 505
VOCAB_LOW = 256
VOCAB_RANK = 16
VOCAB_GROUP_PAD = 512
VOCAB_AFFINE_A = (1, 3, 7, 9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39)
VOCAB_AFFINE_B = tuple((i * 7919) % VOCAB for i in range(VOCAB_RANK))
assert VOCAB == VOCAB_GROUPS * VOCAB_LOW

# Kernel launch config chosen by tune_rpv16_fused_ce.py on the GB10 (sm_121); see the JSON receipt.
# Sweep (4096-row chunk, production sharing the GPU, best-of-15): num_warps 1 -> 13.1 ms, 2 -> 2.3 ms,
# 4 / 8 / 16 -> 0.87-0.89 ms (indistinguishable, ~memory-bandwidth bound); num_stages 1..4 makes no difference.
# Overridable for tuning (mutate the dict, or RPV16_CE_NUM_WARPS / RPV16_CE_NUM_STAGES in the environment).
KERNEL_CONFIG = {"num_warps": int(os.environ.get("RPV16_CE_NUM_WARPS", "4")),
                 "num_stages": int(os.environ.get("RPV16_CE_NUM_STAGES", "2"))}

_AFFINE_CACHE = {}


def _affine_tensors(device):
    key = (device.type, device.index)
    t = _AFFINE_CACHE.get(key)
    if t is None:
        t = (torch.tensor(VOCAB_AFFINE_A, device=device, dtype=torch.long),
             torch.tensor(VOCAB_AFFINE_B, device=device, dtype=torch.long))
        _AFFINE_CACHE[key] = t
    return t


@triton.jit
def _rpv16_ce_row_kernel(
    LOGITS, MIX, TGT, AFF_A, AFF_B, NLL,
    stride_lrow, stride_mrow,
    scale,
    V: tl.constexpr, NG: tl.constexpr, GP: tl.constexpr, LOW: tl.constexpr, R: tl.constexpr,
    WITH_GRAD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, R)
    gi = tl.arange(0, GP)
    li = tl.arange(0, LOW)

    # (i) affine relabel with integer math; python-style remainder like torch.remainder
    tok = tl.load(TGT + row)
    a = tl.load(AFF_A + r)
    b = tl.load(AFF_B + r)
    code = (tok * a + b) % V
    code = (code + V) % V
    tgt_g = code // LOW
    tgt_l = code % LOW

    lbase = LOGITS + row * stride_lrow
    gptr = lbase + r[:, None] * (GP + LOW) + gi[None, :]
    lptr = lbase + r[:, None] * (GP + LOW) + GP + li[None, :]

    # (ii) Match the reference finite padding sentinel, including extreme-logit inputs.
    g = tl.load(gptr).to(tl.float32)
    g = tl.where(gi[None, :] < NG, g, -1.0e9)
    g_hit = gi[None, :] == tgt_g[:, None]
    gmax = tl.max(g, axis=1)
    ge = tl.exp(g - gmax[:, None])
    gsum = tl.sum(ge, axis=1)
    g_t = tl.sum(tl.where(g_hit, g, 0.0), axis=1)
    lg = (g_t - gmax) - tl.log(gsum)

    l = tl.load(lptr).to(tl.float32)
    l_hit = li[None, :] == tgt_l[:, None]
    lmax = tl.max(l, axis=1)
    le = tl.exp(l - lmax[:, None])
    lsum = tl.sum(le, axis=1)
    l_t = tl.sum(tl.where(l_hit, l, 0.0), axis=1)
    lo = (l_t - lmax) - tl.log(lsum)

    mptr = MIX + row * stride_mrow + r
    m = tl.load(mptr).to(tl.float32)
    mmax = tl.max(m, axis=0)
    me = tl.exp(m - mmax)
    msum = tl.sum(me, axis=0)
    lm = (m - mmax) - tl.log(msum)

    # Keep ordinary trained logits on the original FP32 path. Rare extreme
    # logits need wider rank-score accumulation to retain small mixture offsets.
    extreme = (tl.max(tl.abs(g_t - gmax), axis=0) > 1.0e4) | (tl.max(tl.abs(l_t - lmax), axis=0) > 1.0e4) | (tl.max(tl.abs(m - mmax), axis=0) > 1.0e4)
    # Triton requires variables used after a runtime branch to exist on every path.
    # This neutral FP64 scalar is overwritten on the extreme path and unused otherwise.
    base64 = tl.full((), 0.0, tl.float64)
    if extreme:
        t64 = ((g_t.to(tl.float64) - gmax.to(tl.float64)) - tl.log(gsum).to(tl.float64)
               + (l_t.to(tl.float64) - lmax.to(tl.float64)) - tl.log(lsum).to(tl.float64)
               + (m.to(tl.float64) - mmax.to(tl.float64)) - tl.log(msum).to(tl.float64))
        base64 = tl.max(t64, axis=0)
        t = (t64 - base64).to(tl.float32)
    else:
        t = lm + lg + lo
    tmax = tl.max(t, axis=0)
    te = tl.exp(t - tmax)
    tsum = tl.sum(te, axis=0)
    if extreme:
        nll_value = -(base64 + (tmax + tl.log(tsum)).to(tl.float64))
    else:
        # Keep branch result types identical for Triton SSA; store casts to FP32 NLL.
        nll_value = -(tmax + tl.log(tsum)).to(tl.float64)
    tl.store(NLL + row, nll_value)

    # (iii) gradients written in place; scale (= 1/N) applied in FP32 before the cast to the logits dtype
    if WITH_GRAD:
        w = te / tsum                      # posterior over ranks
        ws = w * scale
        dg = ws[:, None] * (ge / gsum[:, None] - tl.where(g_hit, 1.0, 0.0))
        # Reference overwrites padding with a constant, so its input derivative is zero.
        dg = tl.where(gi[None, :] < NG, dg, 0.0)
        tl.store(gptr, dg.to(LOGITS.dtype.element_ty))
        dl = ws[:, None] * (le / lsum[:, None] - tl.where(l_hit, 1.0, 0.0))
        tl.store(lptr, dl.to(LOGITS.dtype.element_ty))
        dm = (me / msum - w) * scale
        tl.store(mptr, dm.to(MIX.dtype.element_ty))


def rpv16_kernel_launch(logits, mix, targets, nll, scale, with_grad):
    """logits [n,12288] and mix [n,16] (row-contiguous, same float dtype) are OVERWRITTEN with grads if with_grad."""
    n = logits.shape[0]
    assert logits.shape[1] == VOCAB_RANK * (VOCAB_GROUP_PAD + VOCAB_LOW) and logits.stride(1) == 1
    assert mix.shape == (n, VOCAB_RANK) and mix.stride(1) == 1
    assert targets.dtype == torch.long and targets.is_contiguous() and targets.shape[0] == n
    assert nll.dtype == torch.float32 and nll.is_contiguous() and nll.shape[0] == n
    aff_a, aff_b = _affine_tensors(logits.device)
    _rpv16_ce_row_kernel[(n,)](
        logits, mix, targets, aff_a, aff_b, nll,
        logits.stride(0), mix.stride(0),
        float(scale),
        V=VOCAB, NG=VOCAB_GROUPS, GP=VOCAB_GROUP_PAD, LOW=VOCAB_LOW, R=VOCAB_RANK,
        WITH_GRAD=bool(with_grad),
        num_warps=KERNEL_CONFIG["num_warps"], num_stages=KERNEL_CONFIG["num_stages"],
    )


class _RPV16FusedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hh, yy, factor_weight, mix_weight, chunk, mean, hint):
        need_h, need_fw, need_mw = ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]
        with_grad = bool(need_h or need_fw or need_mw)
        n, d = hh.shape
        dt = hh.dtype
        dev = hh.device
        nlogit = factor_weight.shape[0]
        scale = (hint / n) if mean else hint       # d(loss_total)/d(nll_row), folded in before the BF16 rounding
        ctx.hint = hint
        chunk = max(1, min(int(chunk), n))
        with torch.cuda.device(dev):
            logits_buf = torch.empty((chunk, nlogit), device=dev, dtype=dt)
            mix_buf = torch.empty((chunk, VOCAB_RANK), device=dev, dtype=dt)
            nll = torch.empty((n,), device=dev, dtype=torch.float32)
            dh = torch.empty_like(hh, memory_format=torch.contiguous_format) if need_h else None
            dfw = torch.zeros_like(factor_weight, memory_format=torch.contiguous_format) if need_fw else None
            dmw = torch.zeros_like(mix_weight, memory_format=torch.contiguous_format) if need_mw else None
            tmp_fw = torch.empty_like(dfw) if need_fw else None
            tmp_mw = torch.empty_like(dmw) if need_mw else None
            tmp_h = torch.empty((chunk, d), device=dev, dtype=dt) if need_h else None
            fw_t = factor_weight.t()
            mw_t = mix_weight.t()
            offs = list(range(0, n, chunk))
            # Reverse chunk order: autograd runs production's per-chunk checkpoint nodes last-chunk-first, so
            # the low-precision dW accumulation order (dW_last + dW_prev + ...) is the same as production's.
            for off in reversed(offs):
                end = min(off + chunk, n)
                m = end - off
                z = hh[off:end]
                lg = logits_buf[:m]
                mx = mix_buf[:m]
                torch.mm(z, fw_t, out=lg)          # same GEMM as F.linear(z, W) for 2-D z, bias=None
                torch.mm(z, mw_t, out=mx)
                rpv16_kernel_launch(lg, mx, yy[off:end], nll[off:end], scale, with_grad)
                if need_h:
                    torch.mm(lg, factor_weight, out=dh[off:end])
                    th = tmp_h[:m]
                    torch.mm(mx, mix_weight, out=th)
                    dh[off:end].add_(th)
                if need_fw:
                    torch.mm(lg.t(), z, out=tmp_fw)
                    dfw.add_(tmp_fw)
                if need_mw:
                    torch.mm(mx.t(), z, out=tmp_mw)
                    dmw.add_(tmp_mw)
            # loss reduction exactly like production: FP32 chunk sums accumulated in forward chunk order
            nll_sum = torch.zeros((), device=dev, dtype=torch.float32)
            for off in offs:
                nll_sum = nll_sum + nll[off:min(off + chunk, n)].sum()
            out = nll_sum / n if mean else nll_sum
        ctx.save_for_backward(dh, dfw, dmw)
        # NLL is already computed by the fused kernel; expose it for detached SAT regret targets.
        ctx.mark_non_differentiable(nll)
        return out, nll

    @staticmethod
    def backward(ctx, grad_out, grad_nll):
        dh, dfw, dmw = ctx.saved_tensors
        g = grad_out.to(torch.float32)
        if ctx.hint != 1.0:
            g = g / ctx.hint                        # == 1.0 when the caller's hint was right

        def sc(x):
            # scale in FP32 then round once (exact no-op for grad_out == 1.0 and any power of two)
            return None if x is None else (x.to(torch.float32) * g).to(x.dtype)

        return sc(dh), None, sc(dfw), sc(dmw), None, None, None


def rpv16_fused_ce(hidden_rows, labels, factor_weight, mix_weight, ce_chunk=4096, reduction="mean",
                   grad_scale_hint=1.0, return_token_nll=False):
    """Rank-16 product-vocab CE, production-equivalent.

    hidden_rows : [N,256] (or [...,256]) float tensor (production: bf16) -- out_proj(norm(x)) rows
    labels      : [N] (or [...]) int64 next-token labels, already shifted; every row counts (no ignore mask,
                  same as production; out-of-range labels wrap like torch.remainder, same as production)
    factor_weight: [16*(512+256), 256] = model.factor_head.weight (no bias in production)
    mix_weight  : [16, 256]            = model.mix_head.weight    (no bias in production)
    returns     : FP32 scalar ce = sum(nll)/N  (reduction="mean", production) or sum(nll) (reduction="sum").
                  With return_token_nll=True, returns (ce, detached FP32 NLL [N]) in original row order.
                  NLL is an auxiliary target/metric only; backpropagate through ce.
    grad_scale_hint: the upstream gradient the caller WILL feed to this loss, i.e. 1/grad_accum for
                  (loss / grad_accum).backward().  Production runs --grad-accum 1, so the default 1.0 is exact.
                  The grads are computed in forward (Liger-style) and rounded to BF16 there; production rounds
                  to BF16 AFTER multiplying by the upstream gradient.  For upstream gradients that are a power
                  of two the two orders are bit-identical; for others (e.g. 1/3) pass the hint so the factor is
                  folded in before the rounding (otherwise a second BF16 rounding adds ~4e-3 relative noise).
                  The result is always mathematically correct: backward rescales by grad_output / hint.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    hh = hidden_rows.reshape(-1, hidden_rows.shape[-1])
    yy = labels.reshape(-1)
    if yy.dtype != torch.long:
        yy = yy.long()
    yy = yy.contiguous()
    if hh.shape[0] != yy.shape[0]:
        raise ValueError(f"rows {hh.shape[0]} != labels {yy.shape[0]}")
    if factor_weight.shape != (VOCAB_RANK * (VOCAB_GROUP_PAD + VOCAB_LOW), hh.shape[1]) or mix_weight.shape != (VOCAB_RANK, hh.shape[1]):
        raise ValueError("unexpected head weight shapes")
    if not (hh.is_cuda and hh.dtype == factor_weight.dtype == mix_weight.dtype):
        raise ValueError("hidden rows and head weights must be CUDA tensors of one dtype")
    if hh.stride(1) != 1:
        hh = hh.contiguous()
    if not torch.is_grad_enabled():
        hh, factor_weight, mix_weight = hh.detach(), factor_weight.detach(), mix_weight.detach()
    hint = float(grad_scale_hint)
    if not (hint > 0.0 and hint < float("inf")):
        raise ValueError("grad_scale_hint must be a positive finite float")
    ce, nll = _RPV16FusedCE.apply(hh, yy, factor_weight, mix_weight, int(ce_chunk), reduction == "mean", hint)
    return (ce, nll) if return_token_nll else ce

# <<< FUSED_CE_EMBED END

# >>> FABLE_FP32_MASTER_V2 BEGIN (fable-cowork-trainers 2026-10-03, v43i "trunk-only, warmed-up fp32 masters")
# Why: RPV16 keeps 99.45M trainable params in bf16 with no fp32 master copy, so AdamW steps below the bf16
# half-ulp round away (attention |w|~0.02-0.04 -> half-ulp 6e-5..1.2e-4, while lr*|m/sqrt(v)| is mostly smaller):
# those weights are de-facto frozen. v43e gave fp32 masters to ALL of them and diverged after ~1.5k updates: the
# head (final RMSNorm weight, exactly 1.0 for 400k steps -> max 1.42 within 1.5k updates, |h| p90 +50%) lost its
# full-depth calibration because local stage objectives decode intermediate depths through the shared head.
# v2 rules: (1) masters ONLY for explicitly registered trunk weights (attention q/k/v/o per stage and in_proj by
# default; routers optional via AGILLM_RPV16_FP32_MASTER_SET); the tied embedding, out_proj, final norm, every
# n1/n2 RMSNorm weight and every other parameter keep the exact v43d/v43h behaviour (passed to the unchanged
# bitsandbytes optimizer untouched). (2) the configured decoupled weight decay applies to every master (all 2-D).
# (3) master updates are warmed up per optimizer: the k-th step of a wrapper applies scale=min(1,(k+1)/WARMUP)
# of the AdamW displacement (Adam moments themselves update normally). (4) AGILLM_RPV16_FP32_MASTER=0 (default)
# restores the stock optimizers exactly (no wrapper object is created at all). Checkpoints stay bf16 with an
# unchanged optimizer-state layout (bnb steps the masters in the same groups/order), so v43d/v43h resume them.
import torch as _fm_torch

RPV16_FP32_MASTER = os.environ.get("AGILLM_RPV16_FP32_MASTER", "0") != "0"
RPV16_FP32_MASTER_WARMUP = max(1, int(os.environ.get("AGILLM_RPV16_FP32_MASTER_WARMUP", "1500") or 1500))
RPV16_FP32_MASTER_SET = os.environ.get("AGILLM_RPV16_FP32_MASTER_SET", "attn,in_proj")
_FM_MASTER_OF = {}          # id(model param) -> (param, fp32 master); the registry pins p so ids never recycle
_FM_ELIGIBLE_IDS = set()    # id(model param) allowed to get a master (empty -> nothing gets a master)
_FM_STATS = {"wrappers": 0, "masters": 0, "master_numel": 0, "resyncs": 0, "steps": 0, "warmup_steps": 0}


def fm_register_eligible(params):
    for p in params:
        if p.dtype in (_fm_torch.bfloat16, _fm_torch.float16) and p.dim() >= 2:
            _FM_ELIGIBLE_IDS.add(id(p))


def fm_has_eligible(params):
    for x in params:
        for p in (x["params"] if isinstance(x, dict) else [x]):
            if id(p) in _FM_ELIGIBLE_IDS:
                return True
    return False


def fm_trunk_params(model):
    """Explicit trunk weights by attribute (never by dtype sweep): attention q/k/v/o, in_proj, optional routers."""
    want = set(x.strip() for x in RPV16_FP32_MASTER_SET.split(",") if x.strip())
    bad = want - {"attn", "router", "in_proj"}
    if bad:
        raise ValueError(f"AGILLM_RPV16_FP32_MASTER_SET has unsupported entries {sorted(bad)}")
    out = []
    for st in model.stages:
        if "attn" in want:
            out += [st.attn.q.weight, st.attn.k.weight, st.attn.v.weight, st.attn.o.weight]
        if "router" in want:
            out.append(st.router.weight)
    if "in_proj" in want:
        out.append(model.in_proj.weight)
    return out


def fm_assert_trunk_only(model):
    """Refuse to train if anything head-side or any RMSNorm weight was registered for a master."""
    forbidden = [model.embed.weight, model.out_proj.weight, model.norm.weight, model.tied_bias]
    for st in model.stages:
        forbidden += [st.n1.weight, st.n2.weight]
    hit = [i for i, p in enumerate(forbidden) if id(p) in _FM_ELIGIBLE_IDS]
    if hit:
        raise RuntimeError(f"fp32-master v2 is trunk-only; forbidden params registered: {hit}")


class _FP32MasterOptimizer:
    """Duck-typed optimizer; param_groups/state/state_dict/load_state_dict/zero_grad/step delegate to `inner`.
    Params without a master are passed to the inner optimizer untouched (identical to the stock optimizer)."""

    def __init__(self, make_inner, params):
        if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
            src_groups = [dict(g) for g in params]
        else:
            src_groups = [{"params": list(params)}]
        self._entries = []   # [model_param, master, group_index] (masters only)
        inner_groups = []
        for gi, g in enumerate(src_groups):
            plist = []
            for p in list(g["params"]):
                m = None
                if id(p) in _FM_ELIGIBLE_IDS and p.dtype in (_fm_torch.bfloat16, _fm_torch.float16):
                    hit = _FM_MASTER_OF.get(id(p))
                    m = hit[1] if (hit is not None and hit[0] is p) else None
                    if m is None:
                        m = _fm_torch.nn.Parameter(p.detach().float().clone(), requires_grad=True)
                        _FM_MASTER_OF[id(p)] = (p, m)
                        _FM_STATS["masters"] += 1
                        _FM_STATS["master_numel"] += int(p.numel())
                    self._entries.append([p, m, gi])
                plist.append(m if m is not None else p)
            ng = dict(g)
            ng["params"] = plist
            inner_groups.append(ng)
        self.inner = make_inner(inner_groups)
        self._synced = False
        self._k = 0
        _FM_STATS["wrappers"] += 1

    @property
    def param_groups(self):
        return self.inner.param_groups

    @property
    def state(self):
        return self.inner.state

    @property
    def defaults(self):
        return self.inner.defaults

    def state_dict(self):
        sd = dict(self.inner.state_dict())
        entries = []
        for p, master, gi in self._entries:
            indexes = [j for j, q in enumerate(self.inner.param_groups[gi]["params"]) if q is master]
            if len(indexes) != 1:
                raise RuntimeError("FP32 master is not unique in its optimizer group")
            entries.append({"group": gi, "param": indexes[0], "value": master.detach()})
        sd["_agillm_fp32_master_v1"] = {
            "schema": 1, "k": int(self._k), "synced": bool(self._synced),
            "warmup_updates": int(RPV16_FP32_MASTER_WARMUP), "entries": entries,
            "legacy_origin": getattr(self, "_legacy_origin", None)}
        return sd

    def load_state_dict(self, sd):
        saved = sd.get("_agillm_fp32_master_v1")
        if saved is None:
            note = getattr(self, "_legacy_restore_note", None)
            if note is None:
                raise RuntimeError("FP32_MASTER_STATE_MISSING: explicit checkpoint-pinned migration required")
            # model.load_state_dict has already restored the BF16 model. Old
            # checkpoints contain no sub-BF16 residuals; do not claim to recover them.
            out = self.inner.load_state_dict(sd)
            self.resync()
            self._k = int(note["k"])
            self._legacy_origin = dict(note)
            self._legacy_restore_note = None
            _FM_STATS["legacy_migrations"] = _FM_STATS.get("legacy_migrations", 0) + 1
            print(json.dumps({"event": "fp32_master_legacy_migrated", **note,
                "restored_master_numel": sum(m.numel() for _p,m,_gi in self._entries),
                "lost_residuals_recovered": False}), flush=True)
            return out
        if saved.get("schema") != 1 or saved.get("warmup_updates") != int(RPV16_FP32_MASTER_WARMUP):
            raise RuntimeError("FP32 master schema/warm-up mismatch")
        if type(saved.get("k")) is not int or saved["k"] < 0 or type(saved.get("synced")) is not bool:
            raise RuntimeError("Invalid FP32 master counter/sync flag")
        if len(saved.get("entries", [])) != len(self._entries):
            raise RuntimeError("FP32 master entry count mismatch")
        pairs = []
        for (p, master, gi), entry in zip(self._entries, saved["entries"]):
            indexes = [j for j, q in enumerate(self.inner.param_groups[gi]["params"]) if q is master]
            value = entry.get("value")
            if (len(indexes) != 1 or entry.get("group") != gi or entry.get("param") != indexes[0]
                    or not _fm_torch.is_tensor(value) or value.shape != master.shape
                    or value.dtype != _fm_torch.float32):
                raise RuntimeError("FP32 master position/shape/dtype mismatch")
            if not bool(_fm_torch.isfinite(value).all()):
                raise RuntimeError("Nonfinite FP32 master checkpoint")
            if saved["synced"] and not _fm_torch.equal(value.to(device=p.device, dtype=p.dtype), p.detach()):
                raise RuntimeError("FP32 master does not round back to its restored model parameter")
            pairs.append((master, value))
        out = self.inner.load_state_dict({k: v for k,v in sd.items() if k != "_agillm_fp32_master_v1"})
        with _fm_torch.no_grad():
            for master, value in pairs:
                master.copy_(value.to(device=master.device))
        self._k = saved["k"]
        self._synced = saved["synced"]
        self._legacy_origin = saved.get("legacy_origin")
        _FM_STATS["exact_restores"] = _FM_STATS.get("exact_restores", 0) + 1
        return out

    def __getattr__(self, name):
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    @_fm_torch.no_grad()
    def resync(self):
        for p, m, _gi in self._entries:
            m.data.copy_(p.data.float())
        self._synced = True
        _FM_STATS["resyncs"] += 1

    def zero_grad(self, set_to_none=True):
        self.inner.zero_grad(set_to_none=set_to_none)   # non-master params (identical to stock); masters below
        for p, m, _gi in self._entries:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()
            m.grad = None

    @_fm_torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("_FP32MasterOptimizer does not support closures")
        if self._entries and not self._synced:
            self.resync()
        scale = min(1.0, float(self._k + 1) / float(RPV16_FP32_MASTER_WARMUP))
        warm = scale < 1.0
        prev = []
        for p, m, _gi in self._entries:
            if p.grad is None:
                m.grad = None
                prev.append(None)
                continue
            m.grad = p.grad.detach().float()
            prev.append(m.detach().clone() if warm else None)   # transient: one optimizer's masters at a time
        out = self.inner.step()
        for (p, m, _gi), pv in zip(self._entries, prev):
            if m.grad is None:
                continue
            if pv is not None:
                m.data.lerp_(pv, 1.0 - scale)                    # m = prev + scale * (m_new - prev)
            p.data.copy_(m.data)
            m.grad = None
        self._k += 1
        _FM_STATS["steps"] += 1
        if warm:
            _FM_STATS["warmup_steps"] += 1
        return out


def _fm_prepare_checkpoint_resume(ck, opts, path, before_stat):
    """Authorize only the explicitly audited legacy file; modern files need no migration."""
    bank = ck.get("optimizer_bank", {}).get("stages", [])
    wrapped = [(i,o) for i,o in enumerate(opts) if isinstance(o, _FP32MasterOptimizer)]
    if not wrapped:
        return
    if len(bank) != len(opts):
        raise RuntimeError("FP32 master resume requires a complete optimizer bank")
    missing = [(i,o) for i,o in wrapped if "_agillm_fp32_master_v1" not in bank[i]]
    if not missing:
        print(json.dumps({"event": "fp32_master_resume_mode", "mode": "exact_saved_masters",
                          "banks": len(wrapped)}), flush=True)
        return
    if len(missing) != len(wrapped):
        raise RuntimeError("Mixed legacy/exact FP32 master bank is not a valid migration")
    policy_path = os.environ.get("AGILLM_RPV16_MASTER_MIGRATION", "")
    if not policy_path:
        raise RuntimeError("Legacy FP32 master checkpoint needs an explicit migration receipt")
    policy = json.loads(Path(policy_path).read_text())
    if policy.get("schema") != "agillm.master-legacy-migration.v1" or policy.get("residual_policy") != "initialize_from_saved_bf16":
        raise RuntimeError("Unsupported FP32 master legacy migration policy")
    for name in ("step", "seen_tokens", "optimizer_update_clock", "target_alignment"):
        if policy.get(name) != ck.get(name):
            raise RuntimeError("Legacy migration checkpoint identity mismatch: " + name)
    if policy.get("warmup_updates") != RPV16_FP32_MASTER_WARMUP:
        raise RuntimeError("Legacy migration warm-up mismatch")
    ks = policy.get("k_by_stage")
    if (not isinstance(ks, list) or len(ks) != len(opts)
            or any(type(k) is not int or k < 0 for k in ks)
            or not policy.get("complete_run_log_verified")):
        raise RuntimeError("Legacy migration lacks complete audited counters")
    def signature(st):
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if signature(os.stat(path)) != signature(before_stat):
        raise RuntimeError("Checkpoint changed during legacy migration load")
    import hashlib as _fm_hashlib
    digest = _fm_hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if signature(os.stat(path)) != signature(before_stat) or digest.hexdigest() != policy.get("checkpoint_sha256"):
        raise RuntimeError("Legacy migration checkpoint checksum mismatch")
    for i,o in missing:
        o._legacy_restore_note = {"bank": i, "k": ks[i], "checkpoint_step": ck["step"],
            "checkpoint_sha256": digest.hexdigest(), "policy_path": policy_path,
            "residual_policy": policy["residual_policy"], "counter_source": "complete_run_log_replay"}
    print(json.dumps({"event": "fp32_master_migration_authorized", "checkpoint_step": ck["step"],
        "banks": len(missing), "k_by_stage": ks, "checkpoint_sha256": digest.hexdigest()}), flush=True)


def fm_unwrap(o):
    return getattr(o, "inner", o)


def fm_telemetry():
    return dict(_FM_STATS, enabled=bool(RPV16_FP32_MASTER), master_bytes=int(_FM_STATS["master_numel"]) * 4,
                eligible=len(_FM_ELIGIBLE_IDS), warmup=RPV16_FP32_MASTER_WARMUP, master_set=RPV16_FP32_MASTER_SET,
                schema_rules="trunk-only-v2")
# <<< FABLE_FP32_MASTER_V2 END


def training_topology(model):
    """Cache references only for the fixed trainer topology; never checkpoint them.

    Model device/dtype transforms and state restoration clear this cache. Call
    clear_training_topology_cache() after intentional module/parameter surgery.
    Parameter order and per-module alias deduplication match nn.Module exactly.
    """
    cache = getattr(model, "_training_topology_cache", None)
    if cache is None:
        modules = tuple(model.modules())
        params = {m: tuple(m.parameters()) for m in modules}
        sparse = tuple(m for m in modules if isinstance(m, SparseLinear))
        cache = {
            "parameters": params,
            "all": params[model],
            "sparse": sparse,
            "stage_sparse": tuple(tuple(m for m in st.modules()
                                        if isinstance(m, SparseLinear))
                                  for st in model.stages),
            # Do not deduplicate across stages: the original telemetry counted
            # a shared parameter once for each stage that contains it.
            "body": tuple((p, p.numel()) for st in model.stages for p in params[st]),
        }
        model._training_topology_cache = cache
        print(json.dumps({"event":"host_topology_cache_enabled",
                          "deployment":"pf2-topology-live-20260920",
                          "parameter_objects":len(cache["all"]),
                          "sparse_layers":len(cache["sparse"]),
                          "source":__file__}), flush=True)
    return cache

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, TOKEN_DIM, device="cuda", dtype=torch.bfloat16)
        self.in_proj = nn.Linear(TOKEN_DIM, D, bias=False, device="cuda", dtype=torch.bfloat16)
        self.stages = nn.ModuleList([Stage(w) for w in WINDOWS])
        self.norm = RMSNorm(D)
        self.out_proj = nn.Linear(D, TOKEN_DIM, bias=False, device="cuda", dtype=torch.bfloat16)
        self.factor_head = nn.Linear(TOKEN_DIM, VOCAB_RANK * (VOCAB_GROUP_PAD + VOCAB_LOW), bias=False, device="cuda", dtype=torch.bfloat16)
        self.mix_head = nn.Linear(TOKEN_DIM, VOCAB_RANK, bias=False, device="cuda", dtype=torch.bfloat16)
        # 2026-09-23 cowork tied classifier (RPV16_TIED_HEAD): weight is self.embed.weight.
        self.tied_bias = nn.Parameter(torch.zeros(VOCAB, device="cuda", dtype=torch.float32))
        self.tied_logit_scale = nn.Parameter(torch.ones((), device="cuda", dtype=torch.float32))
        self.register_buffer("tied_warmup_done", torch.zeros((), device="cuda", dtype=torch.long))
    @torch.no_grad()
    def init_factor_head_from_embedding(self):
        # Warm-start each normalized product component from the trained token
        # embedding under a different bijective affine vocabulary layout.
        # For each rank, E[token] ~= group_weight[g] + low_weight[l].
        E = self.embed.weight.detach().float()
        W = self.factor_head.weight.view(VOCAB_RANK, VOCAB_GROUP_PAD + VOCAB_LOW, TOKEN_DIM)
        W.zero_()
        ids = torch.arange(VOCAB, device=E.device, dtype=torch.long)
        for r,(a,b) in enumerate(zip(VOCAB_AFFINE_A, VOCAB_AFFINE_B)):
            # p(token)=(a*token+b) mod V; scatter trained embeddings into code order.
            code = torch.remainder(ids * int(a) + int(b), VOCAB)
            grid = torch.empty((VOCAB, TOKEN_DIM), device=E.device, dtype=torch.float32)
            grid.index_copy_(0, code, E)
            grid = grid.view(VOCAB_GROUPS, VOCAB_LOW, TOKEN_DIM)
            gw = grid.mean(dim=1)
            lw = (grid - gw[:,None,:]).mean(dim=0)
            W[r, :VOCAB_GROUPS].copy_(gw.to(W.dtype))
            W[r, VOCAB_GROUP_PAD:].copy_(lw.to(W.dtype))
        self.mix_head.weight.zero_()

    def clear_training_topology_cache(self):
        self.__dict__.pop("_training_topology_cache", None)

    def _apply(self, fn, recurse=True):
        self.clear_training_topology_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_training_topology_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def head_parameters(self):
        # Parameters owned by the head optimizer. Tied mode: the classifier weight IS embed.weight, which stays in the
        # stage-0 group (it also carries the input-side gradient); only norm/out_proj/bias/scale belong to the head.
        if RPV16_TIED_HEAD:
            # tied_logit_scale is a calibrated constant (redundant with out_proj), never trained.
            return list(self.norm.parameters()) + list(self.out_proj.parameters()) + [self.tied_bias]
        return (list(self.norm.parameters()) + list(self.out_proj.parameters())
                + list(self.factor_head.parameters()) + list(self.mix_head.parameters()))

    def set_trainable_stage(self, stage_idx=None, head_anchor=False):
        # GB10-native local-update schedule. Exactly one Transformer stage owns
        # gradients on stage updates. A head anchor keeps all stages frozen and
        # updates only the tied token interface, avoiding a 2B-parameter global
        # gradient spike while still training embeddings/input/output/norm.
        topology = training_topology(self)
        for p in topology["all"]:
            p.requires_grad_(False)
        if head_anchor:
            # Output-side tied embedding + final projection only. The Transformer
            # body is evaluated under no_grad during this phase.
            for p in self.head_parameters():
                p.requires_grad_(True)
        elif stage_idx is not None:
            for p in topology["parameters"][self.stages[int(stage_idx)]]:
                p.requires_grad_(True)
            # Stage 0 already requires an end-to-end gradient path, so train the
            # input token interface there at essentially no extra graph depth.
            if int(stage_idx) == 0:
                for mod in (self.embed, self.in_proj):
                    for p in topology["parameters"][mod]: p.requires_grad_(True)

    def set_trainable_joint(self, head=True):
        # 2026-09-23 cowork joint update: every stage and the input token interface own gradients on every update;
        # the output head too unless this update's objective must not touch it (--allheads-head-grad none on SAT/NAT).
        topology = training_topology(self)
        for p in topology["all"]:
            p.requires_grad_(False)
        mods = list(self.stages) + [self.embed, self.in_proj]
        for mod in mods:
            for p in topology["parameters"][mod]:
                p.requires_grad_(True)
        if head:
            for p in self.head_parameters():
                p.requires_grad_(True)

    def forward(self, ids, labels=None, train_stage=None, head_anchor=False):
        # Rotating stage updates use a local tied-token objective: the prefix is
        # frozen, exactly one stage is trainable, and its representation is decoded
        # immediately by the shared final token head. This deliberately removes
        # downstream-stage backward to maximize GB10 sparse-NVFP4 wall-time share.
        aux_total = torch.zeros((), device=ids.device, dtype=torch.float32)
        route_counts = []
        if head_anchor:
            # Head-only phase: do not retain a 16-stage activation graph merely
            # to train the tied output vocabulary interface. The embedding still
            # receives its exact output-side CE gradient because it is the tied
            # LM-head weight below. Input-side embedding/in_proj gradients are
            # handled on the rotating stage-0 phase.
            with torch.no_grad():
                x = self.in_proj(self.embed(ids))
                for j in range(STAGES):
                    x, aux, counts = self.stages[j](x)
                    aux_total = aux_total + aux
                    route_counts.append(counts)
            x = x.detach()
        elif train_stage is None:
            x = self.in_proj(self.embed(ids))
            for j in range(STAGES):
                x, aux, counts = self.stages[j](x)
                aux_total = aux_total + aux
                route_counts.append(counts)
        elif int(train_stage) == JOINT_STAGE:
            # 2026-09-23 cowork joint update: no-autograd full-stack forward; joint_backward() replays the stages.
            x, aux_total, route_counts = joint_trunk(self, ids)
        else:
            # GB10-local objective: frozen prefix, exactly one trainable stage,
            # then jump directly to the shared tied-token decoder. There is no
            # downstream-stage forward/backward on a rotating stage update.
            ts = int(train_stage)
            if ts == 0:
                # Stage 0 also owns the input token interface, so keep this path
                # in autograd. Later stages consume a frozen prefix.
                x = self.in_proj(self.embed(ids))
            else:
                with torch.no_grad():
                    x = self.in_proj(self.embed(ids))
                    for j in range(ts):
                        x, aux, counts = self.stages[j](x)
                        aux_total = aux_total + aux
                        route_counts.append(counts)
            x, aux, counts = self.stages[ts](x)
            aux_total = aux_total + aux
            route_counts.append(counts)
            if RPV16_E2E_SUFFIX and ts < STAGES - 1 and torch.is_grad_enabled():
                # 2026-09-23 cowork E2E credit: carry the trainable stage through the frozen downstream
                # stages (checkpointed, weights never receive grads) so its update follows the gradient of
                # the real full-stack objective. Suffix router aux is not optimised here (stage-owned).
                for j in range(ts + 1, STAGES):
                    x, counts_j = _frozen_suffix_stage(self.stages[j], x)   # V43J FSX (falls back to checkpoint)
                    route_counts.append(counts_j)
        h = self.out_proj(self.norm(x))
        if RPV16_TIED_HEAD and getattr(self, "_tied_needs_calibration", False):
            tied_calibrate(self, h.detach())
        if head_anchor and labels is not None:
            _fh = h.reshape(-1, TOKEN_DIM)
            _fy = labels.reshape(-1)
            _n = min(16, int(_fh.shape[0]))
            _idx = torch.linspace(0, int(_fh.shape[0]) - 1, _n, device=_fh.device).long()
            self._rpv_exact_head_cache = (
                _fh.index_select(0, _idx).detach().clone(),
                _fy.index_select(0, _idx).detach().clone(),
            )
        loss = None
        logits = None
        if labels is not None:
            if RPV16_TIED_HEAD:
                ce = tied_ce(self, h.reshape(-1, TOKEN_DIM), labels.reshape(-1))
            elif getattr(self, "fused_ce", False):
                hh = h.reshape(-1, TOKEN_DIM)
                yy = labels.reshape(-1)
                ce = rpv16_fused_ce(hh, yy, self.factor_head.weight, self.mix_head.weight,
                                    ce_chunk=int(getattr(self, "ce_chunk", 4096)),
                                    grad_scale_hint=float(getattr(self, "fused_ce_grad_scale_hint", 1.0)))
            else:
                # Rank-16 product vocabulary: 129280 == 505 * 256 exactly.
                # This is an exact normalized model distribution (not sampled CE):
                # p(token)=sum_r p(r)*p(group|r)*p(low|r). Seven padded group
                # classes are masked. The 256->12288 projection is GB10-friendly.
                from torch.utils.checkpoint import checkpoint
                # HFStream.batch already supplies labels shifted by one token, so every
                # hidden position i predicts its matching next-token label i.
                hh = h.reshape(-1, TOKEN_DIM)
                yy = labels.reshape(-1)
                affine_a = torch.tensor(VOCAB_AFFINE_A, device=yy.device, dtype=torch.long)
                affine_b = torch.tensor(VOCAB_AFFINE_B, device=yy.device, dtype=torch.long)
                chunk = int(getattr(self, "ce_chunk", 4096))
                nll_sum = torch.zeros((), device=h.device, dtype=torch.float32)
                def rpv_chunk(z, target, factor_weight, mix_weight, aa, bb):
                    raw = F.linear(z, factor_weight).float().view(-1, VOCAB_RANK, VOCAB_GROUP_PAD + VOCAB_LOW)
                    gl = raw[:, :, :VOCAB_GROUP_PAD]
                    gl[:, :, VOCAB_GROUPS:] = -1.0e9
                    ll = raw[:, :, VOCAB_GROUP_PAD:]
                    code = torch.remainder(target[:,None] * aa[None,:] + bb[None,:], VOCAB)
                    target_group = torch.div(code, VOCAB_LOW, rounding_mode="floor")
                    target_low = torch.remainder(code, VOCAB_LOW)
                    lg = gl.log_softmax(-1).gather(2, target_group.unsqueeze(2)).squeeze(2)
                    lo = ll.log_softmax(-1).gather(2, target_low.unsqueeze(2)).squeeze(2)
                    lm = F.linear(z, mix_weight).float().log_softmax(-1)
                    return -torch.logsumexp(lm + lg + lo, dim=1).sum()
                for off in range(0, hh.shape[0], chunk):
                    end = min(off + chunk, hh.shape[0])
                    nll_sum = nll_sum + checkpoint(rpv_chunk, hh[off:end], yy[off:end], self.factor_head.weight, self.mix_head.weight, affine_a, affine_b, use_reentrant=False)
                ce = nll_sum / yy.numel()
            loss = ce + 0.01 * aux_total / STAGES
        else:
            logits = tied_logits(self, h) if RPV16_TIED_HEAD else self.factor_head(h)
        return loss, logits, aux_total / STAGES, route_counts
    def invalidate_sparse(self):
        for m in training_topology(self)["sparse"]:
            m.invalidate()
    def invalidate_stage(self, stage_idx):
        for m in training_topology(self)["stage_sparse"][int(stage_idx)]:
            m.invalidate()
    def sparse_telemetry(self):
        xs=training_topology(self)["sparse"]
        return {"layers":len(xs),"kernel_calls":sum(m.kernel_calls for m in xs),"packs":sum(m.pack_count for m in xs)}


@torch.no_grad()
def _rpv16_frozen_head_nll(model, h, labels):
    """Exact product-vocabulary NLL for already-frozen 256-d head features."""
    hw = h.to(device=model.factor_head.weight.device, dtype=model.factor_head.weight.dtype)
    yy = labels.to(device=hw.device, dtype=torch.long)
    raw = F.linear(hw, model.factor_head.weight).float().view(-1, VOCAB_RANK, VOCAB_GROUP_PAD + VOCAB_LOW)
    gl = raw[:, :, :VOCAB_GROUP_PAD].clone()
    gl[:, :, VOCAB_GROUPS:] = -1.0e9
    ll = raw[:, :, VOCAB_GROUP_PAD:]
    aa = torch.tensor(VOCAB_AFFINE_A, device=yy.device, dtype=torch.long)
    bb = torch.tensor(VOCAB_AFFINE_B, device=yy.device, dtype=torch.long)
    code = torch.remainder(yy[:, None] * aa[None, :] + bb[None, :], VOCAB)
    tg = torch.div(code, VOCAB_LOW, rounding_mode="floor")
    tok_low = torch.remainder(code, VOCAB_LOW)
    lg = gl.log_softmax(-1).gather(2, tg.unsqueeze(2)).squeeze(2)
    lo = ll.log_softmax(-1).gather(2, tok_low.unsqueeze(2)).squeeze(2)
    lm = F.linear(hw, model.mix_head.weight).float().log_softmax(-1)
    return -torch.logsumexp(lm + lg + lo, dim=1)


def _rpv16_cov_apply(p, x):
    return p * x - p * torch.sum(p * x)


def _rpv16_cov_apply_batched(p, x):
    return p * x - p * torch.sum(p * x, dim=-1, keepdim=True)


def _rpv16_scaled_softmax_solve(p, v, scale, damping):
    """Solve (scale*C(p)+damping*I)x=v without dividing by scale."""
    d = damping + scale * p
    den = damping * torch.sum(p / d)
    num = torch.sum(p * v / d)
    return v / d + (scale * p / d) * (num / den)


def _rpv16_scaled_softmax_solve_batched(p, v, scale, damping):
    """Exact batched form of _rpv16_scaled_softmax_solve over the final axis."""
    d = damping + scale * p
    den = damping * torch.sum(p / d, dim=-1, keepdim=True)
    num = torch.sum(p * v / d, dim=-1, keepdim=True)
    return v / d + (scale * p / d) * (num / den)


@torch.no_grad()
def _rpv16_exact_logit_newton_direction(model, h, label, damping=1.0):
    """Exact float64 RPV16 Newton solve using compressed Woodbury + reduced coercivity certificate."""
    damping = float(damping)
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("RPV exact-head damping must be finite and positive")
    hw = h.detach().to(device=model.factor_head.weight.device, dtype=model.factor_head.weight.dtype).reshape(1, TOKEN_DIM)
    raw_gpu = F.linear(hw, model.factor_head.weight).float().view(VOCAB_RANK, VOCAB_GROUP_PAD + VOCAB_LOW)
    mix_gpu = F.linear(hw, model.mix_head.weight).float().reshape(VOCAB_RANK)
    raw = raw_gpu.cpu().double(); mix = mix_gpu.cpu().double()
    group = raw[:, :VOCAB_GROUPS]; low = raw[:, VOCAB_GROUP_PAD:]
    pi = mix.softmax(0); q = group.softmax(1); slo = low.softmax(1)
    target = int(label.detach().cpu())
    aa = torch.tensor(VOCAB_AFFINE_A, dtype=torch.long); bb = torch.tensor(VOCAB_AFFINE_B, dtype=torch.long)
    code = torch.remainder(target * aa + bb, VOCAB)
    tg = torch.div(code, VOCAB_LOW, rounding_mode="floor"); tl = torch.remainder(code, VOCAB_LOW)
    paths = mix.log_softmax(0) + q[torch.arange(VOCAB_RANK), tg].log() + slo[torch.arange(VOCAB_RANK), tl].log()
    alpha = paths.softmax(0)
    gm = pi - alpha
    gg = alpha[:, None] * q; gg[torch.arange(VOCAB_RANK), tg] -= alpha
    gl = alpha[:, None] * slo; gl[torch.arange(VOCAB_RANK), tl] -= alpha
    grad = torch.cat((gm, gg.reshape(-1), gl.reshape(-1))); rhs = -grad
    R, G, L = VOCAB_RANK, VOCAB_GROUPS, VOCAB_LOW
    def unpack(v):
        m=v[:R]; ga=v[R:R+R*G].reshape(R,G); lo=v[R+R*G:].reshape(R,L); return m,ga,lo
    def pack(m,ga,lo): return torch.cat((m,ga.reshape(-1),lo.reshape(-1)))

    Calpha = torch.diag(alpha) - alpha[:,None] * alpha[None,:]
    vg_basis = -q.clone(); vg_basis[torch.arange(R), tg] += 1.0
    vl_basis = -slo.clone(); vl_basis[torch.arange(R), tl] += 1.0
    def binv_apply(v):
        vm,vg,vl=unpack(v)
        ym=_rpv16_scaled_softmax_solve(pi,vm,1.0,damping)
        sc=alpha[:,None]
        return pack(ym,
                    _rpv16_scaled_softmax_solve_batched(q,vg,sc,damping),
                    _rpv16_scaled_softmax_solve_batched(slo,vl,sc,damping))
    y0=binv_apply(rhs); y0m,y0g,y0l=unpack(y0)
    dmix=damping+pi; qmix=damping*torch.sum(pi/dmix); amix=pi/dmix
    mix_inv=torch.diag(1.0/dmix)+torch.outer(amix,amix)/qmix
    sc=alpha[:,None]
    bg=_rpv16_scaled_softmax_solve_batched(q,vg_basis,sc,damping)
    bl=_rpv16_scaled_softmax_solve_batched(slo,vl_basis,sc,damping)
    M=mix_inv+torch.diag(torch.sum(vg_basis*bg,dim=1)+torch.sum(vl_basis*bl,dim=1))
    Vy0=y0m+torch.sum(vg_basis*y0g,dim=1)+torch.sum(vl_basis*y0l,dim=1)
    S=torch.eye(R,dtype=torch.float64)-Calpha@M
    z=torch.linalg.solve(S,Calpha@Vy0)
    correction=binv_apply(pack(z,z[:,None]*vg_basis,z[:,None]*vl_basis)); y=y0+correction

    ym,yg,yl=unpack(y)
    Vy=ym+torch.sum(vg_basis*yg,dim=1)+torch.sum(vl_basis*yl,dim=1)
    c=Calpha@Vy; vt_c=pack(c,c[:,None]*vg_basis,c[:,None]*vl_basis)
    def bop(v):
        vm,vg,vl=unpack(v)
        om=_rpv16_cov_apply(pi,vm)+damping*vm
        sc=alpha[:,None]
        return pack(om,
                    sc*_rpv16_cov_apply_batched(q,vg)+damping*vg,
                    sc*_rpv16_cov_apply_batched(slo,vl)+damping*vl)
    Ay=bop(y)-vt_c
    residual=float((Ay-rhs).norm()/rhs.norm().clamp_min(1e-30))

    sa=torch.sqrt(alpha)
    proj=torch.eye(R,dtype=torch.float64)-torch.outer(sa,sa)
    Lfac=sa[:,None]*proj
    reduced=Lfac.T@M@Lfac; reduced=(reduced+reduced.T)*0.5
    rho=float(torch.linalg.eigvalsh(reduced)[-1]); margin=1.0-rho
    return y, {"relative_logit_operator_residual_f64":residual,
               "small_system_cond_f64":float(torch.linalg.cond(S)),
               "reduced_update_rho_f64":rho,
               "reduced_coercivity_margin_f64":margin,
               "reduced_spd_certified":bool(margin > 0.0),
               "logit_grad_norm_f64":float(grad.norm()),
               "direction_norm_f64":float(y.norm())}


@torch.no_grad()
def rpv16_exact_headlift_correction(model, damping=1.0):
    """Guarded exact single-feature Newton correction for the live RPV output heads."""
    cache=getattr(model,"_rpv_exact_head_cache",None)
    model._rpv_exact_head_cache=None
    if cache is None:
        return {"schema":"agillm.rpv16.exact-headlift.v1","accepted":False,"reason":"no_frozen_head_cache"}
    hs,ys=cache
    hs=hs.detach(); ys=ys.detach()
    h0=hs[0]; y0=ys[0]
    direction,meta=_rpv16_exact_logit_newton_direction(model,h0,y0,damping)
    if not math.isfinite(meta["relative_logit_operator_residual_f64"]) or meta["relative_logit_operator_residual_f64"] > 1e-9:
        return {"schema":"agillm.rpv16.exact-headlift.v1","accepted":False,"reason":"operator_residual","damping":float(damping),**meta}
    before=_rpv16_frozen_head_nll(model,hs,ys).detach().float().cpu()
    old_factor=model.factor_head.weight.detach().clone()
    old_mix=model.mix_head.weight.detach().clone()
    R,G,L=VOCAB_RANK,VOCAB_GROUPS,VOCAB_LOW
    ym=direction[:R]
    yg=direction[R:R+R*G].reshape(R,G)
    yl=direction[R+R*G:].reshape(R,L)
    actual=torch.zeros((R,VOCAB_GROUP_PAD+L),dtype=torch.float64)
    actual[:,:G]=yg; actual[:,VOCAB_GROUP_PAD:]=yl
    s=float(h0.detach().double().cpu().square().sum())
    if not math.isfinite(s) or s <= 0.0:
        return {"schema":"agillm.rpv16.exact-headlift.v1","accepted":False,"reason":"zero_feature_norm","damping":float(damping),**meta}
    hdev=h0.detach().to(device=old_factor.device,dtype=torch.float32)
    df=(actual.reshape(-1).to(device=old_factor.device,dtype=torch.float32)[:,None]*hdev[None,:]/s)
    dm=(ym.to(device=old_mix.device,dtype=torch.float32)[:,None]*hdev[None,:]/s)
    accepted=False; chosen=0.0; after=before; realized_err=float("inf")
    # A line search changes only the step length, never the exact Newton direction itself.
    for scale in (1.0,0.5,0.25,0.125):
        model.factor_head.weight.copy_(old_factor)
        model.mix_head.weight.copy_(old_mix)
        model.factor_head.weight.add_((df*scale).to(model.factor_head.weight.dtype))
        model.mix_head.weight.add_((dm*scale).to(model.mix_head.weight.dtype))
        cand=_rpv16_frozen_head_nll(model,hs,ys).detach().float().cpu()
        if (torch.isfinite(cand).all() and float(cand[0]) <= float(before[0]) + 1e-7
                and float(cand.mean()) <= float(before.mean()) + 1e-7):
            accepted=True; chosen=float(scale); after=cand
            # Measure BF16/weight-rounding realization in logit space on the solved row.
            hw=h0.detach().to(device=model.factor_head.weight.device,dtype=model.factor_head.weight.dtype).reshape(1,TOKEN_DIM)
            raw_after=F.linear(hw,model.factor_head.weight).float().view(R,VOCAB_GROUP_PAD+L).cpu().double()
            mix_after=F.linear(hw,model.mix_head.weight).float().reshape(R).cpu().double()
            model.factor_head.weight.copy_(old_factor); model.mix_head.weight.copy_(old_mix)
            raw_before=F.linear(hw,model.factor_head.weight).float().view(R,VOCAB_GROUP_PAD+L).cpu().double()
            mix_before=F.linear(hw,model.mix_head.weight).float().reshape(R).cpu().double()
            model.factor_head.weight.add_((df*scale).to(model.factor_head.weight.dtype)); model.mix_head.weight.add_((dm*scale).to(model.mix_head.weight.dtype))
            realized=torch.cat((mix_after-mix_before,(raw_after[:,:G]-raw_before[:,:G]).reshape(-1),(raw_after[:,VOCAB_GROUP_PAD:]-raw_before[:,VOCAB_GROUP_PAD:]).reshape(-1)))
            target=direction*scale
            realized_err=float((realized-target).norm()/target.norm().clamp_min(1e-30))
            break
    if not accepted:
        model.factor_head.weight.copy_(old_factor); model.mix_head.weight.copy_(old_mix)
        after=before
    return {"schema":"agillm.rpv16.exact-headlift.v1","accepted":bool(accepted),"damping":float(damping),
            "step_scale":chosen,"validation_rows":int(hs.shape[0]),"target_nll_before":float(before[0]),
            "target_nll_after":float(after[0]),"validation_mean_nll_before":float(before.mean()),
            "validation_mean_nll_after":float(after.mean()),"feature_norm_sq_f64":s,
            "bf16_realized_direction_relative_error":realized_err,**meta}


_LOG_LOCK = threading.Lock()

def _emit(obj):
    # v2 runs a data producer thread next to the loop. print() issues the text and the
    # newline as two writes; emit each JSON line as one locked write instead. Bytes on the
    # wire are identical to print(json.dumps(obj), flush=True).
    line = json.dumps(obj) + "\n"
    with _LOG_LOCK:
        sys.stdout.write(line); sys.stdout.flush()


class PrefetchAborted(Exception):
    """A stop signal arrived while the loop was starved for data; no batch was consumed."""


class HFStream:
    """Restart-exact streaming corpus mixer.

    Hugging Face IterableDataset.state_dict() resumes the underlying stream, but
    HF's built-in shuffle intentionally drops its in-memory shuffle buffer on
    resume. For a 600B-token run that makes every restart silently alter/replay
    corpus coverage. We therefore keep the small row shuffle buffer ourselves and
    checkpoint it alongside the underlying dataset state, source RNG, and leftover
    token buffer.
    """
    STATE_SCHEMA = "agillm.gb10.1pf.data-cursor.v1"
    ROW_SHUFFLE = 200

    def __init__(self, names, seed=42):
        from datasets import load_dataset
        from tokenizers import Tokenizer
        self.load_dataset = load_dataset
        raw = TOKENIZER_JSON.read_text()
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("tokenizer_bundle"), dict) and isinstance(obj["tokenizer_bundle"].get("tokenizer.json"), str):
            self.tok = Tokenizer.from_str(obj["tokenizer_bundle"]["tokenizer.json"])
        else:
            self.tok = Tokenizer.from_str(raw)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.sources = [n for n in names if n in HF_SOURCES]
        if not self.sources: raise ValueError("no valid HF sources")
        self.weights = [HF_SOURCES[n][2] for n in self.sources]
        self.datasets = {}
        self.iters = {}
        self.row_buffers = {}
        self.disabled = set()
        self.buffer = []
        self.rows_emitted = {n: 0 for n in self.sources}
        # v2 prefetch (inert until start_prefetch): producer thread, bounded queue, and the
        # cursor snapshot belonging to the last batch the training loop actually consumed.
        self._pf_thread = None
        self._pf_queue = None
        self._pf_stop = None
        self._consumed = None
        self._loaded_dataset_state = None  # HF cursor handed to load_state_dict(); see start_prefetch()
        # Per-source exact iterator state used only to reconstruct a stream after a
        # transient remote FileNotFoundError / expired signed URL. This is not a
        # second cursor: after every successful yield it is replaced with the
        # dataset's own current state.
        self._transport_resume_state = {}
        self.abort_check = None
        self.last_qsize = None
        self._pf_gilfree = False
        self._pf_verify = 0

    def _open(self, name):
        repo,cfg,_=HF_SOURCES[name]
        kw=dict(split="train",streaming=True)
        if name == "fineweb-edu":
            kw["revision"] = FINEWEB_EDU_REVISION
        ds=self.load_dataset(repo,cfg,**kw) if cfg else self.load_dataset(repo,**kw)
        # Do NOT call HF IterableDataset.shuffle here: its shuffle buffer is not
        # included in state_dict(). Our explicit row_buffers below are.
        self.datasets[name]=ds
        self.iters[name]=iter(ds)
        return self.iters[name]

    def _reopen_at_transport_state(self, name, state):
        """Rebuild one HF iterator at an exact datasets state.

        Used only after a transport-resolution failure.  The source is reopened
        at the pinned revision, then the saved iterator state is installed before
        any row is consumed.
        """
        self.iters.pop(name, None)
        self.datasets.pop(name, None)
        self.disabled.discard(name)
        self._open(name)
        if state is not None:
            ds=self.datasets[name]
            if not hasattr(ds, "load_state_dict"):
                raise RuntimeError(f"stream source {name} lacks load_state_dict() during transport retry")
            ds.load_state_dict(copy.deepcopy(state))
            self.iters[name]=iter(ds)

    def _text(self,row):
        for k in ("text","content","code"):
            v=row.get(k) if isinstance(row,dict) else None
            if isinstance(v,str) and v.strip(): return v
        return ""

    def _raw_ids(self,name):
        # Finite sample configs eventually hit StopIteration and intentionally
        # wrap.  A remote FileNotFoundError is different: FineWeb's Xet/LFS
        # object still exists, but a signed URL / fsspec resolution may fail
        # transiently.  In that case rebuild the SAME iterator state and retry.
        wraps = 0
        transport_retries = 0
        while True:
            if name not in self.iters:
                self._open(name)
            ds=self.datasets.get(name)
            retry_state=self._transport_resume_state.get(name)
            if retry_state is None and ds is not None and hasattr(ds,"state_dict"):
                retry_state=copy.deepcopy(ds.state_dict())
            try:
                row=next(self.iters[name])
            except FileNotFoundError as e:
                transport_retries += 1
                if transport_retries > 5:
                    raise
                delay=min(8.0,0.5*(2**(transport_retries-1)))
                _emit({"event":"hf_transport_retry","source":name,
                       "attempt":transport_retries,"delay_s":delay,
                       "rows_emitted":int(self.rows_emitted.get(name,0)),
                       "error":type(e).__name__,"detail":str(e)[:240],
                       "cursor_action":"reopen_same_revision_same_state"})
                time.sleep(delay)
                self._reopen_at_transport_state(name,retry_state)
                continue
            except StopIteration:
                wraps += 1
                if wraps > 3:
                    raise RuntimeError(f"HF source {name} still empty after {wraps} wraps")
                _emit({"event":"hf_source_wrap","source":name,"wrap":wraps,
                       "rows_emitted":int(self.rows_emitted.get(name,0)),
                       "reason":"StopIteration on finite streaming sample; reopening from start"})
                self._transport_resume_state.pop(name,None)
                self.iters.pop(name, None)
                self.datasets.pop(name, None)
                self.disabled.discard(name)
                self._open(name)
                continue
            # Successful next(): datasets state is now authoritative for a retry
            # of the following row/chunk.
            transport_retries = 0
            ds=self.datasets.get(name)
            if ds is not None and hasattr(ds,"state_dict"):
                self._transport_resume_state[name]=copy.deepcopy(ds.state_dict())
            text=self._text(row)
            if not text:
                continue
            ids=self._encode_ids(text) if self._pf_gilfree else self.tok.encode(text).ids
            if len(ids)>4095:
                ids=ids[:4095]
            if ids:
                return ids

    def _prime_rows(self,name):
        buf=self.row_buffers.setdefault(name,[])
        while len(buf)<self.ROW_SHUFFLE:
            try: buf.append(self._raw_ids(name))
            except StopIteration: break
        return buf

    def _next_ids_from_source(self,name):
        buf=self._prime_rows(name)
        if not buf: raise StopIteration
        j=self.rng.randrange(len(buf))
        out=buf[j]
        try:
            buf[j]=self._raw_ids(name)
        except StopIteration:
            buf.pop(j)
        self.rows_emitted[name]=self.rows_emitted.get(name,0)+1
        return out

    def next_ids(self):
        available=[n for n in self.sources if n not in self.disabled]
        if not available: raise RuntimeError("all HF sources disabled")
        for _ in range(max(4,len(available)*2)):
            # v56: honor bound self.weights (quality_mix active.json); skip non-positive
            wmap={src:w for src,w in zip(self.sources,self.weights)} if getattr(self,"weights",None) else {}
            pairs=[(n, float(wmap.get(n, HF_SOURCES[n][2]))) for n in available]
            pairs=[(n,w) for n,w in pairs if w > 0]
            if not pairs:
                raise RuntimeError("all HF sources have non-positive weights")
            available_w, weights = zip(*pairs)
            n=self.rng.choices(list(available_w),weights=list(weights),k=1)[0]
            try:
                return n,self._next_ids_from_source(n)
            except StopIteration:
                # Prefer wrap/reopen (finite sample configs) over permanent disable.
                try:
                    self.iters.pop(n,None); self.datasets.pop(n,None); self.disabled.discard(n)
                    self._open(n)
                    _emit({"event":"hf_source_wrap","source":n,"wrap":"next_ids",
                           "rows_emitted":int(self.rows_emitted.get(n,0)),
                           "reason":"StopIteration in next_ids; reopened from start"})
                    return n, self._next_ids_from_source(n)
                except Exception as wrap_e:
                    _emit({"event":"hf_source_wrap_failed","source":n,"error":type(wrap_e).__name__,"detail":str(wrap_e)[:240]})
                    self.disabled.add(n); self.iters.pop(n,None); self.datasets.pop(n,None)
                    available=[x for x in available if x!=n]
                    if not available: raise RuntimeError("all HF sources exhausted")
            except Exception as e:
                _emit({"dataset_disable":n,"error":type(e).__name__,"detail":str(e)[:240]})
                self.disabled.add(n); self.iters.pop(n,None); self.datasets.pop(n,None)
                available=[x for x in available if x!=n]
                if not available: raise
        raise RuntimeError("could not fetch nonempty HF row")

    @staticmethod
    def _pack_rows(rows):
        lens=torch.tensor([len(x) for x in rows],dtype=torch.int32)
        flat=torch.tensor([v for x in rows for v in x],dtype=torch.int32) if rows else torch.empty(0,dtype=torch.int32)
        return {"lengths":lens,"tokens":flat}

    @staticmethod
    def _unpack_rows(obj):
        lens=[int(x) for x in obj.get("lengths",torch.empty(0,dtype=torch.int32)).tolist()]
        flat=obj.get("tokens",torch.empty(0,dtype=torch.int32)).tolist()
        out=[]; off=0
        for n in lens:
            out.append(flat[off:off+n]); off+=n
        if off!=len(flat): raise RuntimeError("corrupt data cursor row buffer")
        return out

    def state_dict(self):
        ds_state={}
        for name,ds in self.datasets.items():
            if not hasattr(ds,"state_dict"):
                raise RuntimeError(f"stream source {name} lacks state_dict()")
            ds_state[name]=ds.state_dict()
        return {
            "schema":self.STATE_SCHEMA,
            "seed":self.seed,
            "sources":list(self.sources),
            "rng_state":self.rng.getstate(),
            "disabled":sorted(self.disabled),
            "token_buffer":torch.tensor(self.buffer,dtype=torch.int32),
            "row_buffers":{n:self._pack_rows(rows) for n,rows in self.row_buffers.items()},
            "dataset_state":ds_state,
            "rows_emitted":dict(self.rows_emitted),
        }

    def load_state_dict(self,state):
        if state.get("schema")!=self.STATE_SCHEMA:
            raise RuntimeError(f"unsupported data cursor schema {state.get('schema')!r}")
        if list(state.get("sources",[]))!=list(self.sources):
            raise RuntimeError(f"data source mismatch checkpoint={state.get('sources')} runtime={self.sources}")
        self.rng.setstate(state["rng_state"])
        self.disabled=set(state.get("disabled",[]))
        tb=state.get("token_buffer",torch.empty(0,dtype=torch.int32))
        self.buffer=[int(x) for x in tb.tolist()]
        self.row_buffers={n:self._unpack_rows(v) for n,v in state.get("row_buffers",{}).items()}
        self.rows_emitted={n:int(v) for n,v in state.get("rows_emitted",{}).items()}
        self.datasets={}; self.iters={}
        import copy  # function-local on purpose: this hunk needs no module-level import
        self._loaded_dataset_state=copy.deepcopy(state.get("dataset_state",{}))  # v2: used by start_prefetch() only
        self._transport_resume_state=copy.deepcopy(state.get("dataset_state",{}))
        for name,ds_state in state.get("dataset_state",{}).items():
            if name in self.disabled: continue
            self._open(name)
            ds=self.datasets[name]
            if not hasattr(ds,"load_state_dict"):
                raise RuntimeError(f"stream source {name} lacks load_state_dict()")
            ds.load_state_dict(ds_state)
            self.iters[name]=iter(ds)

    # ---- v2 prefetch: background producer + consumed-cursor snapshots -------------------
    DEVICE="cuda"

    def _snapshot(self):
        # Cheap capture of the restart-exact cursor at the producer's current position. Row
        # lists are never mutated in place (only replaced or popped), so shallow copies of the
        # row buffers are exact; HF state_dict() already returns a deep copy.
        ds_state={}
        for name,ds in self.datasets.items():
            if not hasattr(ds,"state_dict"):
                raise RuntimeError(f"stream source {name} lacks state_dict()")
            ds_state[name]=ds.state_dict()
        return {"rng_state":self.rng.getstate(),"disabled":sorted(self.disabled),"buffer":list(self.buffer),
                "row_buffers":{n:list(rows) for n,rows in self.row_buffers.items()},
                "dataset_state":ds_state,"rows_emitted":dict(self.rows_emitted)}

    def _materialize(self,snap):
        # Same keys, order, dtypes and values as state_dict() taken at the snapshot position.
        return {
            "schema":self.STATE_SCHEMA,
            "seed":self.seed,
            "sources":list(self.sources),
            "rng_state":snap["rng_state"],
            "disabled":snap["disabled"],
            "token_buffer":torch.tensor(snap["buffer"],dtype=torch.int32),
            "row_buffers":{n:self._pack_rows(rows) for n,rows in snap["row_buffers"].items()},
            "dataset_state":snap["dataset_state"],
            "rows_emitted":snap["rows_emitted"],
        }

    def consumed_state_dict(self):
        """Cursor as of the last batch handed to the training loop, never the read-ahead position."""
        if self._pf_thread is None:
            return self.state_dict()
        return self._materialize(self._consumed)

    GILFREE_VERIFY_ROWS=256

    def _encode_ids(self,text):
        # Producer thread only. Tokenizer.encode() holds the GIL for the whole Rust call (measured
        # on the box with the production tokenizer: a thread that needs the GIL back drops from
        # ~19,000 to ~400 reacquires/s next to it, ~5 ms per reacquire), so a producer using it
        # would stall the kernel-launch thread for most of the host work prefetch is meant to hide.
        # encode_batch([text]) runs the same pipeline on one input with the GIL released. It is
        # never trusted blindly: the first GILFREE_VERIFY_ROWS rows are encoded both ways, and any
        # difference or error switches this process back to encode() for good. The ids that are
        # used are v1's encode() ids in every such case.
        try:
            ids=self.tok.encode_batch([text])[0].ids
        except Exception as e:
            self._pf_gilfree=False
            _emit({"event":"prefetch_gilfree_disabled","ts":time.time(),"reason":"encode_batch failed: "+type(e).__name__})
            return self.tok.encode(text).ids
        if self._pf_verify>0:
            self._pf_verify-=1
            ref=self.tok.encode(text).ids
            if ids!=ref:
                self._pf_gilfree=False
                _emit({"event":"prefetch_gilfree_disabled","ts":time.time(),"reason":"encode_batch ids differ from encode ids; using encode()"})
                return ref
        return ids

    def start_prefetch(self,batch,seq,depth=3,pin=True,gil_release=True):
        if self._pf_thread is not None: raise RuntimeError("prefetch already started")
        self._pf_shape=(int(batch),int(seq)); self._pf_pin=bool(pin)
        self._pf_gilfree=bool(gil_release) and hasattr(self.tok,"encode_batch")
        self._pf_verify=self.GILFREE_VERIFY_ROWS
        if self._pf_gilfree: os.environ.setdefault("TOKENIZERS_PARALLELISM","false")  # one input per call: no rayon pool wanted
        self._pf_queue=queue.Queue(maxsize=max(1,int(depth)))
        self._pf_stop=threading.Event()
        self._consumed=self._snapshot()  # nothing consumed yet: the loaded/initial cursor
        if self._loaded_dataset_state is not None:
            # real `datasets` reports a ROW-0 cursor between load_state_dict() and the first next(): keep the loaded one
            import copy
            self._consumed["dataset_state"]=copy.deepcopy(self._loaded_dataset_state)
        self._pf_thread=threading.Thread(target=self._pf_run,name="hfstream-prefetch",daemon=True)
        self._pf_thread.start()

    def _pf_run(self):
        # Sole owner of the stream state from here on. One producer + FIFO queue keeps the v1
        # batch order; each batch travels with the cursor snapshot taken right after it.
        batch,seq=self._pf_shape
        need=batch*(seq+1)
        try:
            while not self._pf_stop.is_set():
                source_counts={}
                while len(self.buffer)<need:
                    src,ids=self.next_ids(); source_counts[src]=source_counts.get(src,0)+1
                    self.buffer.extend(ids); self.buffer.append(1)
                chunk=self.buffer[:need]; del self.buffer[:need]
                t=torch.tensor(chunk,dtype=torch.long).view(batch,seq+1)
                if self._pf_pin:
                    try: t=t.pin_memory()
                    except Exception: self._pf_pin=False
                self._pf_queue.put(("batch",t,source_counts,self._snapshot()))  # blocks while full
        except BaseException as e:
            self._pf_queue.put(("error",e,None,None))  # surfaces in the main thread, in order

    def _pf_next(self,batch,seq):
        if (int(batch),int(seq))!=self._pf_shape: raise RuntimeError("prefetch batch geometry changed")
        self.last_qsize=self._pf_queue.qsize()
        while True:
            try:
                kind,t,source_counts,snap=self._pf_queue.get(timeout=1.0)
                break
            except queue.Empty:
                if not self._pf_thread.is_alive() and self._pf_queue.empty():
                    raise RuntimeError("data prefetch producer exited without a result")
                if self.abort_check is not None and self.abort_check():
                    raise PrefetchAborted()
        if kind=="error": raise t
        self._consumed=snap
        t=t.to(self.DEVICE,non_blocking=True)
        return t[:,:seq],t[:,1:],source_counts

    def stop_prefetch(self,timeout=5.0):
        th=self._pf_thread
        if th is None or self._pf_stop.is_set(): return
        self._pf_stop.set()
        try:
            while True: self._pf_queue.get_nowait()  # release a producer blocked on a full queue
        except queue.Empty:
            pass
        th.join(timeout)  # daemon thread: a producer stuck in a network read cannot block exit

    def batch(self,batch,seq):
        if self._pf_thread is not None:
            return self._pf_next(batch,seq)
        need=batch*(seq+1)
        source_counts={}
        while len(self.buffer)<need:
            src,ids=self.next_ids(); source_counts[src]=source_counts.get(src,0)+1
            self.buffer.extend(ids); self.buffer.append(1)
        chunk=self.buffer[:need]; del self.buffer[:need]
        t=torch.tensor(chunk,dtype=torch.long,device="cuda").view(batch,seq+1)
        return t[:,:seq],t[:,1:],source_counts


import hashlib
# RPV_DATA_ONLY_QUALITY_MIX_20260924. No model/optimizer/objective code changes.
_QUALITY_ROOT = Path('/workspace/quality_mix_20260924')
_QUALITY_ACTIVE = _QUALITY_ROOT / 'active.json'
_QUALITY_NAMES = {'quality-finemath': 'finemath4', 'quality-python': 'code_selfoss', 'quality-constraints': 'instruction_constraints'}
_QUALITY_ORDER = ['fineweb-edu', *_QUALITY_NAMES]
for _qn, _source in _QUALITY_NAMES.items():
    HF_SOURCES[_qn] = ('json', str(_QUALITY_ROOT / (_source + '.train.jsonl')), 0.08 if _qn == 'quality-constraints' else 0.16)
_OriginalQualityHFStream = HFStream

class HFStream(_OriginalQualityHFStream):
    """Data-only extension with explicit additive migration and preserved consumed cursor."""
    def __init__(self, names, seed=42):
        self.quality_manifest_sha256 = None
        self.quality_manifest = None
        self.quality_files = {}
        self.quality_weights = {}
        names = list(names)
        if _QUALITY_ACTIVE.exists():
            active = json.loads(_QUALITY_ACTIVE.read_text())
            if active.get('schema') != 'agillm.quality_mix.active.v1':
                raise RuntimeError('unsupported quality dataset activation schema')
            if active.get('enabled'):
                manifest_path = Path(active['manifest_path'])
                raw = manifest_path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest != active['manifest_sha256']:
                    raise RuntimeError('quality dataset manifest hash mismatch')
                manifest = json.loads(raw)
                if manifest.get('schema') != 'agillm.quality_mix.v1':
                    raise RuntimeError('unsupported quality corpus manifest schema')
                source_map = {s['name']: s for s in manifest['sources']}
                for name, source in _QUALITY_NAMES.items():
                    rec = source_map[source]['train']
                    p = Path(rec['path'])
                    if p.resolve().parent != _QUALITY_ROOT.resolve() or '.holdout.' in p.name:
                        raise RuntimeError('quality training source outside immutable train directory')
                    h = hashlib.sha256()
                    with p.open('rb') as f:
                        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
                    if h.hexdigest() != rec['sha256'] or p.stat().st_size != rec['bytes']:
                        raise RuntimeError('quality training file content mismatch: '+name)
                    if int(rec['rows']) < 1: raise RuntimeError('empty quality training source')
                    self.quality_files[name] = str(p)
                weights = active.get('rpv_source_weights', {'fineweb-edu':1.6,'quality-finemath':0.16,'quality-python':0.16,'quality-constraints':0.08})
                if set(weights) != set(_QUALITY_ORDER): raise RuntimeError('quality weight source mismatch')
                for name, value in weights.items():
                    value = float(value)
                    if not math.isfinite(value) or value < 0: raise RuntimeError('invalid quality source weight')
                    self.quality_weights[name] = value
                if self.quality_weights['fineweb-edu'] <= 0: raise RuntimeError('base language source must remain active')
                self.quality_manifest_sha256 = digest
                self.quality_manifest = manifest
                if names == ['fineweb-edu']: names = list(_QUALITY_ORDER)
        if any(n in _QUALITY_NAMES for n in names) and self.quality_manifest is None:
            raise RuntimeError('quality sources require an active verified manifest')
        super().__init__(names, seed)
        if self.quality_manifest is not None:
            self.weights = [self.quality_weights.get(n,HF_SOURCES[n][2]) for n in self.sources]
            # Bind tokenizer identity to the original token IDs used in filtering.
            if hashlib.sha256(TOKENIZER_JSON.read_bytes()).hexdigest() != self.quality_manifest['tokenizer_sha256']:
                raise RuntimeError('quality corpus tokenizer does not match trainer tokenizer')
            _emit({'event':'quality_mix_ready','sources':self.sources,'weights':self.weights,
                   'manifest_sha256':self.quality_manifest_sha256,
                   'training_rows':self.quality_manifest['train_rows'],
                   'training_tokens_no_special':self.quality_manifest['train_tokens_no_special']})

    def _open(self, name):
        if name not in _QUALITY_NAMES:
            return super()._open(name)
        ds = self.load_dataset('json', data_files={'train':self.quality_files[name]}, split='train', streaming=True)
        self.datasets[name] = ds
        self.iters[name] = iter(ds)
        return self.iters[name]

    def _quality_bind_state(self, state):
        if self.quality_manifest_sha256:
            state['quality_manifest_sha256'] = self.quality_manifest_sha256
            state['quality_source_weights'] = {n:w for n,w in zip(self.sources,self.weights)}
        return state

    def state_dict(self):
        return self._quality_bind_state(super().state_dict())

    def _materialize(self, snap):
        return self._quality_bind_state(super()._materialize(snap))

    def load_state_dict(self, state):
        old_sources = list(state.get('sources',[]))
        if old_sources != self.sources:
            # Replacement-host exact projection: checkpoint 246907 carries the
            # three quality cursors historically, but all three are explicitly
            # zero-weight. The live selector filters non-positive weights before
            # random.choices(), so the future selected source is fineweb-edu only.
            # Preserve its HF cursor, explicit row shuffle buffer, token buffer,
            # and RNG state; discard only unreachable zero-weight source cursors.
            qweights = state.get('quality_source_weights')
            if (old_sources == _QUALITY_ORDER and self.sources == ['fineweb-edu']
                    and isinstance(qweights, dict)
                    and set(qweights) == set(_QUALITY_ORDER)
                    and float(qweights.get('fineweb-edu', 0.0)) > 0.0
                    and all(float(qweights.get(n, -1.0)) == 0.0 for n in _QUALITY_NAMES)):
                updated = dict(state)
                updated['sources'] = ['fineweb-edu']
                updated['disabled'] = [n for n in state.get('disabled', []) if n == 'fineweb-edu']
                updated['row_buffers'] = {'fineweb-edu': state.get('row_buffers', {}).get('fineweb-edu', {})}
                updated['dataset_state'] = {'fineweb-edu': state.get('dataset_state', {}).get('fineweb-edu', {})}
                updated['rows_emitted'] = {'fineweb-edu': int(state.get('rows_emitted', {}).get('fineweb-edu', 0))}
                # Base cursor loader ignores quality metadata, but remove it from
                # the projected state so future checkpoints describe reality.
                updated.pop('quality_manifest_sha256', None)
                updated.pop('quality_source_weights', None)
                super().load_state_dict(updated)
                _emit({'event':'quality_mix_zero_weight_cursor_projection',
                       'from_sources':old_sources,'to_sources':['fineweb-edu'],
                       'reason':'all_removed_sources_saved_with_exact_zero_weight',
                       'rng_state_preserved':True,'token_buffer_preserved':True,
                       'fineweb_dataset_state_preserved':True,'fineweb_row_buffer_preserved':True})
                return
            if old_sources != ['fineweb-edu'] or self.sources != _QUALITY_ORDER or not self.quality_manifest_sha256:
                raise RuntimeError('unsupported source change; refusing to reset data cursor')
            updated = dict(state)
            updated['sources'] = list(self.sources)
            updated['rows_emitted'] = dict(state.get('rows_emitted',{}))
            for name in _QUALITY_NAMES: updated['rows_emitted'].setdefault(name,0)
            updated['quality_manifest_sha256'] = self.quality_manifest_sha256
            # Existing HF cursor, RNG, shuffle reservoir and leftover tokens are
            # passed unchanged to the original exact-resume implementation.
            super().load_state_dict(updated)
            _emit({'event':'quality_mix_additive_cursor_migration',
                   'from_sources':old_sources,'to_sources':self.sources,
                   'existing_rows_preserved':state.get('rows_emitted',{}),
                   'leftover_tokens_preserved':len(self.buffer),
                   'old_hf_cursor_preserved':True,'shuffle_buffer_preserved':True,'rng_state_preserved':True,
                   'manifest_sha256':self.quality_manifest_sha256})
            return
        expected = state.get('quality_manifest_sha256')
        if self.quality_manifest_sha256 != expected:
            raise RuntimeError('quality manifest changed across resume; explicit migration required')
        super().load_state_dict(state)



# >>> ALLHEADS BEGIN  (generated by make_v3_allheads.py -- do not hand-edit)
# AR + SAT-fixed + SAT-var + NAT on the RPV16 DBlock-stagewise trainer.
#
# Core contract: NO new BF16/FP32 GEMM is added to the trainable transformer core.  The only new
# trainable tensors live OUTSIDE the core in AllHeadsAux: shift_emb [2,256] (an add on the 256-d head
# input) and the SAT-var regret MLP 256->256->2 (fp32, DETACHED input).  SAT adds one extra
# flash-attn call per stage (attention kernel, not a GEMM).  Model/Stage/Expert are byte-identical to v1.
ALLHEADS_SCHEMA = "agillm.gb10.1pf.allheads.v3"
ALLHEADS_CKPT_SCHEMA = "agillm.gb10.1pf.recovery.v6"
ALLHEADS_OBJECTIVES = ("ar", "sat", "nat")
# V89_FLOOR005: allow AR0.985 (sat/nat 0.0075) to bind; prior AR0.985 was noop under floor 0.01
ALLHEADS_PROB_FLOOR = 0.005
NAT_MASK_ID = 2                      # <pad> of the production tokenizer; same id as natmaskedhead_v2 / S
SATVAR_KMAX = 2                      # owner contract: SAT-var chooses between 1 and 2 tokens only
SATVAR_REGRET_HIDDEN = 256
SATVAR_REGRET_PRIOR_CE = (6.3, 6.9)  # S: SATVAR_REGRET_PRIOR_CE[:2]
ALLHEADS_HOT_DEFAULTS = {
    "dblock_ar_prob": 0.60, "dblock_sat_prob": 0.25, "dblock_nat_prob": 0.15,
    "sat_fixed_share": 0.5, "satvar_block_probs": "1:0.5,2:0.5",
    "satvar_lambda": 1.0, "satvar_merge_cost": None, "dblock_satvar_regret_weight": 0.05,
    "satvar_log_every": 20, "satvar_regret_max_blocks": 4096,
    "nat_mask_ratio": "uniform:0.1:1.0", "nat_span_mask_prob": 0.35, "nat_suffix_mask_prob": 0.20,
    "nat_region_partial_prob": 0.5,
}
_AH_M64 = (1 << 64) - 1


def _ah_mix64(x):
    x = (int(x) + 0x9E3779B97F4A7C15) & _AH_M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _AH_M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _AH_M64
    return x ^ (x >> 31)


def ah_hash64(seed, clock, stream=0, micro=0):
    """Counter-based 64-bit hash of (seed, update clock, stream, micro). No stored RNG state."""
    z = _ah_mix64(int(seed) & _AH_M64)
    z = _ah_mix64(z ^ (int(clock) & _AH_M64))
    z = _ah_mix64(z ^ ((int(stream) & 0xFFFFFFFF) << 32) ^ (int(micro) & 0xFFFFFFFF))
    return z


def ah_u01(seed, clock, stream=0, micro=0):
    return (ah_hash64(seed, clock, stream, micro) >> 11) / float(1 << 53)


def ah_generator(seed, clock, stream, micro=0):
    g = torch.Generator(device="cpu")
    g.manual_seed(ah_hash64(seed, clock, stream, micro) & ((1 << 63) - 1))
    return g


def ah_floor_probs(raw, floor=ALLHEADS_PROB_FLOOR):
    """Normalise objective probabilities and enforce p >= floor for EVERY objective (water-filling)."""
    names = ALLHEADS_OBJECTIVES
    vals = []
    for n in names:
        try:
            v = float(raw.get(n, 0.0))
        except Exception:
            v = 0.0
        vals.append(v if (v == v and v > 0.0 and v != float("inf")) else 0.0)
    if sum(vals) <= 0.0:
        vals = [1.0] * len(names)
    fixed = set()
    p = {}
    for _ in range(len(names) + 1):
        free = [i for i in range(len(names)) if i not in fixed]
        mass = 1.0 - floor * len(fixed)
        s = sum(vals[i] for i in free)
        p = {i: floor for i in fixed}
        for i in free:
            p[i] = (vals[i] / s * mass) if s > 0.0 else mass / len(free)
        low = [i for i in free if p[i] < floor - 1e-12]
        if not low:
            break
        fixed.update(low)
    return {names[i]: float(p[i]) for i in range(len(names))}


def ah_draw_objective(seed, clock, probs):
    """Deterministic objective for optimizer update `clock`: pure function of (seed, clock, probs)."""
    u = ah_u01(seed, clock, 0)
    acc = 0.0
    for n in ALLHEADS_OBJECTIVES:
        acc += float(probs[n])
        if u < acc:
            return n
    return ALLHEADS_OBJECTIVES[-1]


def ah_parse_block_probs(spec, floor=ALLHEADS_PROB_FLOOR):
    """'1:0.5,2:0.5' -> P(block size 2); both sizes are floored so neither stride can be configured away."""
    p = {1: 0.0, 2: 0.0}
    try:
        for part in str(spec).split(","):
            k, v = part.split(":")
            k = int(k)
            if k in p:
                p[k] = max(0.0, float(v))
    except Exception:
        p = {1: 0.5, 2: 0.5}
    tot = p[1] + p[2]
    p2 = 0.5 if tot <= 0.0 else p[2] / tot
    return min(1.0 - floor, max(floor, p2))


def ah_parse_mask_rate(spec):
    """'uniform:lo:hi' (default, per-sequence sampled rate) or a fixed float."""
    s = str(spec).strip().lower()
    try:
        if s.startswith("uniform"):
            parts = s.split(":")
            lo = float(parts[1]) if len(parts) > 1 else 0.1
            hi = float(parts[2]) if len(parts) > 2 else 1.0
        else:
            lo = hi = float(s)
    except Exception:
        lo, hi = 0.1, 1.0
    lo = min(1.0, max(0.01, lo)); hi = min(1.0, max(lo, hi))
    return lo, hi


# ---------------------------------------------------------------- SAT partition + attention
def ah_sat_partition(B, T, p2, fixed2, generator):
    """Per-row random partition of T positions into blocks of size {1,2} (CPU tensors).

    Returns first2 [B,T] bool (position is the FIRST token of a size-2 block), slot [B,T] (0/1),
    bsize [B,T] (1/2), bend [B,T] (last position of the block).  fixed2 -> all-2 blocks (fixed SAT).
    """
    B = int(B); T = int(T)
    if fixed2:
        sizes = torch.full((B, T), 2, dtype=torch.long)
    else:
        sizes = 1 + (torch.rand((B, T), generator=generator) < float(p2)).to(torch.long)
    starts = torch.cumsum(sizes, 1) - sizes
    ok = starts < T
    eff = torch.minimum(sizes, T - starts)
    rows = torch.arange(B).view(B, 1).expand(B, T)
    size_at = torch.zeros((B, T), dtype=torch.long)
    size_at[rows[ok], starts[ok]] = eff[ok]
    first2 = size_at == 2
    second2 = torch.zeros((B, T), dtype=torch.bool)
    second2[:, 1:] = first2[:, :-1]
    slot = second2.to(torch.long)
    bsize = 1 + (first2 | second2).to(torch.long)
    bend = torch.arange(T).view(1, T) + first2.to(torch.long)
    return first2, slot, bsize, bend


def ah_routing_alignment(B, T):
    """The balanced affine router couples the 6 tokens at flat indices g + k*(B*T/6). When B*T/6 is a multiple of
    T (production 24x2048) those are the SAME position in different rows, which cannot carry a row's own future
    back to it. Any other coupled geometry can, so block-causal / bidirectional objectives refuse to run on it."""
    n = int(B) * int(T)
    if n % EXPERTS:
        return "per_token"
    return "aligned" if (n // EXPERTS) % int(T) == 0 else "misaligned"


def _ah_sdpa_masked(q, k, v, allow):
    """Dense-mask reference attention. q [B,S,Hq,Dh], k/v [B,Sk,Hkv,Dh], allow [B or 1,S,Sk] bool."""
    hq, hkv = q.shape[2], k.shape[2]
    qq = q.transpose(1, 2)
    kk = k.transpose(1, 2); vv = v.transpose(1, 2)
    if hq != hkv:
        kk = kk.repeat_interleave(hq // hkv, dim=1); vv = vv.repeat_interleave(hq // hkv, dim=1)
    y = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=allow[:, None, :, :])
    return y.transpose(1, 2)


def ah_flash_semantics_dense(q, k, v, window, causal=True):
    """Dense emulation of flash_attn_func(causal, window_size=(window,0)) INCLUDING its bottom-right
    alignment when seqlen_k != seqlen_q: query i sees keys j in [i+Sk-Sq-window, i+Sk-Sq]."""
    S, Sk = q.shape[1], k.shape[1]
    i = torch.arange(S, device=q.device).view(S, 1) + (Sk - S)
    j = torch.arange(Sk, device=q.device).view(1, Sk)
    allow = torch.ones((S, Sk), dtype=torch.bool, device=q.device)
    if causal:
        allow = allow & (j <= i)
    if window is not None and int(window) >= 0:
        allow = allow & (j >= i - int(window))
    return _ah_sdpa_masked(q, k, v, allow[None])


def ah_block_causal_true_dense(q, k, v, window, bend):
    """TRUE partition block-causal attention: q at i attends k at j iff j <= block_end(i)
    (bidirectional inside a block, causal across blocks) and, for sliding-window stages, i-j <= window."""
    S = q.shape[1]
    i = torch.arange(S, device=q.device).view(1, S, 1)
    j = torch.arange(S, device=q.device).view(1, 1, S)
    allow = j <= bend.to(q.device).view(-1, S, 1)
    if window is not None and int(window) >= 0:
        allow = allow & ((i - j) <= int(window))
    return _ah_sdpa_masked(q, k, v, allow)


def ah_block_causal_twocall(q, k, v, window, first2, attn_fn):
    """Two-call block-causal attention for {1,2} partitions, exact at sequence edges and under windows.

    call A = the ordinary causal call (stage window w).
    call B = the same causal call against K/V given ONE extra trailing slot, so the (bottom-right
             aligned) causal frontier of every query moves one key to the right; window w+1 keeps the
             left edge where call A has it.  A query that is the FIRST token of a size-2 block takes
             call B (it sees exactly one extra key: its block partner), everything else takes call A.
    The literal "shift K/V left by one" variant drops key 0 (and key i-w under a window) from call B;
    padding on the right instead is the edge-exact form of the same idea.  The pad slot is only
    visible to the last query, which can never be the first token of a size-2 block.
    """
    w = -1 if window is None else int(window)
    yA = attn_fn(q, k, v, w)
    kB = F.pad(k, (0, 0, 0, 0, 0, 1)); vB = F.pad(v, (0, 0, 0, 0, 0, 1))
    yB = attn_fn(q, kB, vB, -1 if w < 0 else w + 1)
    return torch.where(first2.to(q.device)[:, :, None, None], yB, yA)


def _ah_flash_causal(q, k, v, w):
    from flash_attn import flash_attn_func
    return flash_attn_func(q, k, v, dropout_p=0.0, causal=True, window_size=((-1, -1) if w < 0 else (w, 0)))


class _AHAttnState:
    mode = "ar"          # "ar" | "sat" | "nat_bidir"
    first2 = None


_AH_ATTN = _AHAttnState()


class ah_attn_mode:
    def __init__(self, mode, first2=None):
        self.mode, self.first2 = mode, first2
    def __enter__(self):
        _AH_ATTN.mode, _AH_ATTN.first2 = self.mode, self.first2
        return self
    def __exit__(self, *exc):
        _AH_ATTN.mode, _AH_ATTN.first2 = "ar", None
        return False


def ah_attention(q, k, v, window):
    """Non-AR attention; SAT uses pinned, reversible one-key merge dispatch."""
    mode = _AH_ATTN.mode
    if mode == "sat":
        fn = _ah_flash_causal if q.is_cuda else (lambda a, b, c, w: ah_flash_semantics_dense(a, b, c, w))
        from sg_sat_partner_0ac2c32d import dispatch
        return dispatch(q, k, v, window, _AH_ATTN.first2,
                        lambda: ah_block_causal_twocall(q, k, v, window, _AH_ATTN.first2, fn),
                        emit=_emit)
    if mode == "nat_bidir":
        if q.is_cuda:
            from flash_attn import flash_attn_func
            return flash_attn_func(q, k, v, dropout_p=0.0, causal=False, window_size=(-1, -1))
        return ah_flash_semantics_dense(q, k, v, -1, causal=False)
    raise ValueError(f"unknown attention mode {mode!r}")


# ---------------------------------------------------------------- NAT corruption
def _ah_nat_span_len(T, rate, u):
    target = max(1, min(T, int(round(T * max(0.01, min(0.95, float(rate)))))))
    hi = min(T, max(target, target * 2))
    lo = max(1, min(hi, target // 2 if target > 1 else 1))
    return lo + min(hi - lo, int(float(u) * (hi - lo + 1)))


def ah_nat_corrupt(ids, cfg, generator):
    """S-style corruption mix (suffix / span / Bernoulli) with a per-sequence SAMPLED mask rate so
    iterative mask-predict ("discrete diffusion") decoding is in-distribution.  The mask never
    depends on token values.  Returns (corrupted ids, mask [B,T] bool on ids.device, stats)."""
    B, T = ids.shape
    lo, hi = ah_parse_mask_rate(cfg["nat_mask_ratio"])
    suffix_p = min(1.0, max(0.0, float(cfg["nat_suffix_mask_prob"])))
    span_p = min(1.0 - suffix_p, max(0.0, float(cfg["nat_span_mask_prob"])))
    partial_p = min(1.0, max(0.0, float(cfg["nat_region_partial_prob"])))
    u = torch.rand((B, 6), generator=generator)
    bern = torch.rand((B, T), generator=generator)
    mask = torch.zeros((B, T), dtype=torch.bool)
    kinds = {"suffix": 0, "span": 0, "bernoulli": 0}
    rates = []
    for b in range(B):
        rate = lo + (hi - lo) * float(u[b, 1])
        kind = float(u[b, 0])
        if kind < suffix_p or kind < suffix_p + span_p:
            n = _ah_nat_span_len(T, rate, u[b, 2])
            start = (T - n) if kind < suffix_p else min(T - n, int(float(u[b, 3]) * (T - n + 1)))
            kinds["suffix" if kind < suffix_p else "span"] += 1
            region = torch.zeros(T, dtype=torch.bool); region[start:start + n] = True
            if float(u[b, 4]) < partial_p:
                inner = lo + (hi - lo) * float(u[b, 5])      # a partially refined region (later passes)
                region = region & (bern[b] < inner)
            mask[b] = region
        else:
            kinds["bernoulli"] += 1
            mask[b] = bern[b] < rate
        if not bool(mask[b].any()):
            mask[b, min(T - 1, int(float(u[b, 3]) * T))] = True
        rates.append(rate)
    dmask = mask.to(ids.device)
    corrupted = torch.where(dmask, torch.full_like(ids, int(NAT_MASK_ID)), ids)
    stats = {"mask_frac": float(mask.float().mean()), "kinds": kinds,
             "rate_min": min(rates), "rate_max": max(rates)}
    return corrupted, dmask, stats


# ---------------------------------------------------------------- aux heads (outside the core)
class AllHeadsAux(nn.Module):
    """shift_emb: learned per-slot shift added to the 256-d head input (zero-init: slot 0 == AR head).
    stride_regret: SAT-var policy head regressing log per-slot CE from the block's last hidden (detached)."""
    def __init__(self, device):
        super().__init__()
        self.shift_emb = nn.Parameter(torch.zeros((SATVAR_KMAX, TOKEN_DIM), device=device, dtype=torch.float32))
        self.stride_regret = nn.Sequential(
            nn.Linear(TOKEN_DIM, SATVAR_REGRET_HIDDEN), nn.GELU(), nn.Linear(SATVAR_REGRET_HIDDEN, SATVAR_KMAX)).to(device=device, dtype=torch.float32)
        self.register_buffer("regret_updates", torch.zeros((), dtype=torch.long, device=device), persistent=True)
        self.init_()

    @torch.no_grad()
    def init_(self):
        # deterministic fresh init on a private generator: never consumes the global torch RNG
        self.shift_emb.zero_()
        lin0, lin1 = self.stride_regret[0], self.stride_regret[2]
        g = torch.Generator(device="cpu").manual_seed(20260912)
        bound = 1.0 / math.sqrt(float(TOKEN_DIM))
        lin0.weight.copy_(((torch.rand(lin0.weight.shape, generator=g) * 2.0 - 1.0) * bound).to(lin0.weight.device))
        lin0.bias.zero_()
        lin1.weight.zero_()
        lin1.bias.copy_(torch.log(torch.tensor(SATVAR_REGRET_PRIOR_CE, dtype=torch.float32)).to(lin1.bias.device))
        self.regret_updates.zero_()

    def slot_hidden(self, h, slot):
        return h + self.shift_emb.index_select(0, slot).to(h.dtype)

    def regret_log_ce(self, h_last):
        return self.stride_regret(h_last.detach().float()).float()


def ah_rpv_token_nll(model, hh, yy):
    """Per-token NLL under the exact rank-16 product-vocab distribution (same math as v1 rpv_chunk)."""
    from torch.utils.checkpoint import checkpoint
    affine_a = torch.tensor(VOCAB_AFFINE_A, device=yy.device, dtype=torch.long)
    affine_b = torch.tensor(VOCAB_AFFINE_B, device=yy.device, dtype=torch.long)
    chunk = int(getattr(model, "ce_chunk", 4096))
    def rpv_chunk_tok(z, target, factor_weight, mix_weight, aa, bb):
        raw = F.linear(z, factor_weight).float().view(-1, VOCAB_RANK, VOCAB_GROUP_PAD + VOCAB_LOW)
        gl = raw[:, :, :VOCAB_GROUP_PAD].clone()
        gl[:, :, VOCAB_GROUPS:] = -1.0e9
        ll = raw[:, :, VOCAB_GROUP_PAD:]
        code = torch.remainder(target[:, None] * aa[None, :] + bb[None, :], VOCAB)
        target_group = torch.div(code, VOCAB_LOW, rounding_mode="floor")
        target_low = torch.remainder(code, VOCAB_LOW)
        lg = gl.log_softmax(-1).gather(2, target_group.unsqueeze(2)).squeeze(2)
        lo = ll.log_softmax(-1).gather(2, target_low.unsqueeze(2)).squeeze(2)
        lm = F.linear(z, mix_weight).float().log_softmax(-1)
        return -torch.logsumexp(lm + lg + lo, dim=1)
    outs = []
    for off in range(0, hh.shape[0], chunk):
        end = min(off + chunk, hh.shape[0])
        outs.append(checkpoint(rpv_chunk_tok, hh[off:end], yy[off:end], model.factor_head.weight, model.mix_head.weight, affine_a, affine_b, use_reentrant=False))
    return torch.cat(outs) if len(outs) != 1 else outs[0]


# ============================================================================ tied classifier (cowork 2026-09-23)
class _TiedFusedCE(torch.autograd.Function):
    """Chunked full-vocabulary softmax CE for logits = (h @ E^T) * scale + bias. At most one [chunk, V] block exists
    at a time: forward keeps only the per-row logsumexp, backward recomputes each block. Returns (mean CE, per-row NLL);
    the NLL output is non-differentiable (SAT regret targets)."""

    @staticmethod
    def forward(ctx, h, y, E, bias, scale, chunk):
        N = h.shape[0]
        s = scale.detach().float()
        lse = torch.empty(N, device=h.device, dtype=torch.float32)
        tgt = torch.empty(N, device=h.device, dtype=torch.float32)
        Et = E.t()
        for a in range(0, N, chunk):
            b = min(N, a + chunk)
            z = torch.matmul(h[a:b], Et).float()
            z.mul_(s).add_(bias)
            lse[a:b] = torch.logsumexp(z, dim=1)
            tgt[a:b] = z.gather(1, y[a:b].unsqueeze(1)).squeeze(1)
            del z
        nll = lse - tgt
        ctx.save_for_backward(h, y, E, bias, scale, lse)
        ctx.chunk = int(chunk)
        ctx.mark_non_differentiable(nll)
        return nll.mean(), nll

    @staticmethod
    def backward(ctx, g_ce, g_nll):
        h, y, E, bias, scale, lse = ctx.saved_tensors
        N = h.shape[0]
        chunk = ctx.chunk
        s = scale.detach().float()
        gs = g_ce.float() / N
        need_h, _, need_E, need_b, need_s, _ = ctx.needs_input_grad
        dh = torch.empty_like(h) if need_h else None
        dE = torch.zeros(E.shape, device=E.device, dtype=torch.float32) if need_E else None
        db = torch.zeros(bias.shape, device=bias.device, dtype=torch.float32) if need_b else None
        ds = torch.zeros((), device=h.device, dtype=torch.float32) if need_s else None
        Et = E.t()
        for a in range(0, N, chunk):
            b = min(N, a + chunk)
            raw = torch.matmul(h[a:b], Et).float()
            p = raw * s
            p.add_(bias).sub_(lse[a:b].unsqueeze(1)).exp_()
            p[torch.arange(b - a, device=p.device), y[a:b]] -= 1.0
            p.mul_(gs)                                   # dL/dlogits
            if ds is not None:
                ds += torch.dot(p.reshape(-1), raw.reshape(-1))
            del raw
            if db is not None:
                db += p.sum(0)
            p.mul_(s)                                    # dL/d(h @ E^T)
            pb = p.to(h.dtype)
            del p
            if dh is not None:
                dh[a:b] = torch.matmul(pb, E)
            if dE is not None:
                dE += torch.matmul(pb.t(), h[a:b]).float()
            del pb
        return (dh, None,
                None if dE is None else dE.to(E.dtype),
                None if db is None else db.to(bias.dtype),
                None if ds is None else ds.to(scale.dtype),
                None)


def _tied_scale_value(model):
    s = getattr(model, "_tied_scale_f", None)
    if s is None:
        s = float(model.tied_logit_scale)
        model._tied_scale_f = s
    return s


def tied_ce(model, hh, yy, return_token_nll=False):
    """Tied full-vocab CE, logits = (h @ E^T) * s + b with a calibrated constant s. When the head is frozen (SAT/NAT
    under --allheads-head-grad none) the classifier weight and bias are detached, so those objectives never move the
    output side. Default path: Cut-Cross-Entropy with the bias carried as an extra embedding column
    (e' = [h*s, 1, 0..0], c' = [E, b, 0..0], width TOKEN_DIM+16)."""
    head = bool(model.tied_bias.requires_grad)
    E = model.embed.weight if head else model.embed.weight.detach()
    b = model.tied_bias if head else model.tied_bias.detach()
    s = _tied_scale_value(model)
    if TIED_USE_CCE and _CCE_LCE is not None and hh.is_cuda:
        if TIED_CCE_AFFINE_BIAS and _CCE_BIAS_LCE is not None:
            # Lean-checked real identity: <[s*h,1,0^15],[E,b,0^15]> = s<h,E>+b.
            # The specialized Triton path keeps the classifier at TOKEN_DIM=256 and computes dbias directly.
            e = (hh * s).contiguous()
            nll = _CCE_BIAS_LCE(e, E.contiguous(), b, yy)
        else:
            N = hh.shape[0]
            e = torch.cat([hh * s, torch.ones(N, 1, device=hh.device, dtype=hh.dtype),
                           torch.zeros(N, 15, device=hh.device, dtype=hh.dtype)], 1)
            c = torch.cat([E, b.to(E.dtype).unsqueeze(1), torch.zeros(E.shape[0], 15, device=E.device, dtype=E.dtype)], 1)
            nll = _CCE_LCE(e, c, yy, reduction="none", filter_eps=None)
        ce = nll.float().mean()
        return (ce, nll.detach().float()) if return_token_nll else ce
    ce, nll = _TiedFusedCE.apply(hh, yy, E, b, model.tied_logit_scale.detach(), TIED_CE_CHUNK)
    return (ce, nll) if return_token_nll else ce


def tied_logits(model, h):
    return F.linear(h, model.embed.weight).float() * model.tied_logit_scale + model.tied_bias


@torch.no_grad()
def tied_calibrate(model, h, rows=4096):
    """One-time init when switching from the RPV16 head: logit scale = 1/std of the raw tied logits on real features,
    so the first distribution is the unigram bias plus unit-variance feature logits."""
    hh = h.reshape(-1, TOKEN_DIM)[:rows]
    raw = F.linear(hh, model.embed.weight).float()
    sd = float(raw.std())
    model.tied_logit_scale.fill_(1.0 / max(sd, 1e-6))
    model._tied_scale_f = float(model.tied_logit_scale)
    model._tied_needs_calibration = False
    print(json.dumps({"event": "tied_head_calibrated", "raw_logit_std": round(sd, 5),
                      "scale": round(1.0 / max(sd, 1e-6), 6), "rows": int(hh.shape[0])}), flush=True)


# ============================================================================ joint update (cowork 2026-09-23)
class _JointTape:
    """Stage inputs of one no-autograd full-stack forward, replayed stage by stage by joint_backward()."""
    __slots__ = ("xs", "ys", "auxs", "ids", "attn", "leaf", "aux_w")

    def __init__(self):
        self.clear()

    def clear(self):
        self.xs = None
        self.ys = None
        self.auxs = None
        self.ids = None
        self.attn = ("ar", None)
        self.leaf = None
        self.aux_w = 0.0


_JOINT_TAPE = _JointTape()


# Bounded replay elimination: no change to loss, parameters, objectives or optimizer order.
_BR_POLICY_PATH = Path("/workspace/astra_bounded_retention_20260923/policy.json")
_BR_LAST_COUNT = 0
_BR_LAST_REPORT = None
_BR_MEMORY_LATCH = False

def _bounded_retention_count():
    global _BR_MEMORY_LATCH, _BR_LAST_COUNT, _BR_LAST_REPORT
    wanted = 0
    revision = None
    try:
        cfg = json.loads(_BR_POLICY_PATH.read_text())
        if not isinstance(cfg, dict):
            raise ValueError("retention policy must be an object")
        wanted = cfg.get("max_tail_stages", 0)
        if type(wanted) is not int or not 0 <= wanted <= 4:
            raise ValueError("max_tail_stages must be an integer from 0 through 4")
        revision = cfg.get("revision")
    except (OSError, ValueError, TypeError):
        wanted = 0
    # Check actual shared-system headroom at an update boundary, before building a new graph.
    try:
        with open("/proc/meminfo") as mf:
            available = next(int(line.split()[1]) / 1048576 for line in mf if line.startswith("MemAvailable:"))
    except (OSError, StopIteration, ValueError):
        available = 0.0
    if wanted and available < 14.0 and not _BR_MEMORY_LATCH:
        _BR_MEMORY_LATCH = True
        # Releases only this process's unused cached allocations. Does not stop either trainer.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    count = 0 if _BR_MEMORY_LATCH else min(wanted, max(0, STAGES - 1))
    _BR_LAST_COUNT = count
    report = (wanted, count, _BR_MEMORY_LATCH, revision)
    if report != _BR_LAST_REPORT:
        print(json.dumps({"event":"bounded_retention_config", "requested_stages":wanted,
                          "retained_stages":count, "memory_latched_off":_BR_MEMORY_LATCH,
                          "mem_available_gib":round(available,3), "floor_gib":14.0,
                          "revision":revision, "ts":time.time()}), flush=True)
        _BR_LAST_REPORT = report
    return count


def joint_trunk(model, ids, aux_weight=None):
    """Full 16-stage forward without autograd. Returns (leaf, aux_total, route_counts): the head is applied to
    `leaf` (a detached trunk output that requires grad), so loss.backward() stops there and leaves dL/d(trunk) in
    leaf.grad; joint_backward() then carries it through the stages. The attention mode active NOW (SAT block-causal
    first2 / NAT bidir / AR) is recorded and re-entered for the replay."""
    tape = _JOINT_TAPE
    if tape.leaf is not None:
        tape.clear()
        raise RuntimeError("joint_trunk: the previous joint forward was never replayed (joint mode needs --grad-accum 1)")
    aux_total = torch.zeros((), device=ids.device, dtype=torch.float32)
    route_counts = []
    xs = [None] * STAGES
    ys = [None] * STAGES
    auxs = [None] * STAGES
    retain = _bounded_retention_count()
    _mds_v38.refresh_policy()  # v38: fused down-dgrad GEMM + SiLU epilogue
    _fsb_refresh_policy()   # exact fallback fused SiLU-down backward
    _cmw_refresh_policy()   # exact masked-weight-gradient fusion
    first_retained = STAGES - retain
    with torch.no_grad():
        x = model.in_proj(model.embed(ids))
    for j in range(STAGES):
        if j >= first_retained:
            # Separate graphs preserve replay's stage-by-stage gradient/optimizer ordering.
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                y, aux, counts = model.stages[j](x_in)
            xs[j], ys[j], auxs[j] = x_in, y, aux
            x = y.detach()
            aux_total = aux_total + aux.detach()
        else:
            with torch.no_grad():
                if j > 0:
                    xs[j] = x
                x, aux, counts = model.stages[j](x)
                aux_total = aux_total + aux
        route_counts.append(counts)
    leaf = x.detach().requires_grad_(True)
    tape.xs, tape.ys, tape.auxs = xs, ys, auxs
    tape.ids = ids
    tape.attn = (_AH_ATTN.mode, _AH_ATTN.first2)
    tape.leaf = leaf
    tape.aux_w = float(0.01 / STAGES if aux_weight is None else aux_weight)
    return leaf, aux_total, route_counts


def joint_backward(model, stage_update):
    """Replay stages STAGES-1..0 from their saved inputs, backpropagating dL/d(trunk output) plus each stage's
    router-aux weight (the 0.01*aux/STAGES term of every objective). stage_update(j) is called as soon as stage j's
    parameter gradients are complete (stage 0 includes embed/in_proj); stage j-1's input gradient has already been
    computed with stage j's pre-update weights at that point, so the update is the exact joint gradient step."""
    tape = _JOINT_TAPE
    leaf = tape.leaf
    if leaf is None:
        raise RuntimeError("joint_backward without a pending joint forward")
    g = leaf.grad
    tape.leaf = None
    if g is None:
        tape.clear()
        raise RuntimeError("joint_backward: the loss produced no gradient for the trunk output")
    prev = (_AH_ATTN.mode, _AH_ATTN.first2)
    _AH_ATTN.mode, _AH_ATTN.first2 = tape.attn
    try:
        with torch.enable_grad():
            for j in range(STAGES - 1, -1, -1):
                if tape.ys[j] is not None:
                    x_in, y, aux_j = tape.xs[j], tape.ys[j], tape.auxs[j]
                else:
                    if j == 0:
                        x_in = model.in_proj(model.embed(tape.ids))
                    else:
                        x_in = tape.xs[j].detach().requires_grad_(True)
                    y, aux_j, _ = model.stages[j](x_in)
                outs, grads = [y], [g]
                if tape.aux_w != 0.0 and torch.is_tensor(aux_j) and aux_j.requires_grad:
                    outs.append(aux_j)
                    grads.append(torch.full_like(aux_j, tape.aux_w))
                torch.autograd.backward(outs, grads)
                g = x_in.grad if j > 0 else None
                tape.xs[j] = None
                tape.ys[j] = None
                tape.auxs[j] = None
                del x_in, y, aux_j, outs, grads
                stage_update(j)
    finally:
        _AH_ATTN.mode, _AH_ATTN.first2 = prev
        tape.clear()


def ah_trunk(model, ids, ts):
    """v1's rotating-stage branch verbatim: frozen no_grad prefix 0..ts-1, trainable stage ts, no downstream."""
    if int(ts) == JOINT_STAGE:
        return joint_trunk(model, ids)      # 2026-09-23 cowork joint update (attention mode captured for replay)
    aux_total = torch.zeros((), device=ids.device, dtype=torch.float32)
    route_counts = []
    ts = int(ts)
    if ts == 0:
        x = model.in_proj(model.embed(ids))
    else:
        with torch.no_grad():
            x = model.in_proj(model.embed(ids))
            for j in range(ts):
                x, aux, counts = model.stages[j](x)
                aux_total = aux_total + aux
                route_counts.append(counts)
    x, aux, counts = model.stages[ts](x)
    aux_total = aux_total + aux
    route_counts.append(counts)
    return x, aux_total, route_counts


def ah_satvar_gate_stats(ce1, ce2, pred_log, lam, mu):
    """Gate telemetry on size-2 blocks. Teacher: stride 2 acceptable iff lam - max(0, CE2-CE1) > 0 on the
    REALISED per-slot CE; policy: the same rule on the regret head's CEhat.  Base rate and lift are always
    reported next to precision/recall."""
    gap = (ce2 - ce1).float()
    teacher = (float(lam) - gap.clamp_min(0.0)) > 0.0
    chat = pred_log.float().exp()
    gap_hat = chat[:, 1] - chat[:, 0]
    policy = (float(lam) - gap_hat.clamp_min(0.0)) > 0.0
    policy_merge = (float(lam) - gap_hat.clamp_min(0.0) - float(mu)) > 0.0
    n = int(gap.numel())
    tp = int((teacher & policy).sum()); npos = int(teacher.sum()); npred = int(policy.sum())
    base = npos / max(1, n)
    prec = (tp / npred) if npred > 0 else None
    rec = (tp / npos) if npos > 0 else None
    lift = (prec / base) if (prec is not None and base > 0.0) else None
    qs = torch.quantile(gap[: min(n, 4_000_000)], torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=gap.device)).tolist() if n > 0 else []
    return {"n_blocks2": n, "lam": float(lam), "mu": float(mu),
            "policy_stride_hist": {"1": n - npred, "2": npred},
            "teacher_stride_hist": {"1": n - npos, "2": npos},
            "policy_decision_basis": "learned_ce_regret_plus_explicit_speed_reward",
            "policy_counts_are_emitted_tokens": False,
            "base_rate": round(base, 5), "pred_rate": round(npred / max(1, n), 5),
            "precision": None if prec is None else round(prec, 5), "recall": None if rec is None else round(rec, 5),
            "lift": None if lift is None else round(lift, 4),
            "ce_gap_q10_25_50_75_90": [round(float(x), 4) for x in qs],
            "mean_stride_teacher": round(1.0 + base, 5),
            "mean_stride_policy": round(1.0 + npred / max(1, n), 5),
            "mean_stride_policy_if_merge_needed": round(1.0 + int(policy_merge.sum()) / max(1, n), 5),
            "ce1_mean": round(float(ce1.float().mean()), 4), "ce2_mean": round(float(ce2.float().mean()), 4),
            "cehat1_mean": round(float(chat[:, 0].mean()), 4), "cehat2_mean": round(float(chat[:, 1].mean()), 4)}


def ah_fused_ce_fn(model):
    """The composed stack embeds rpv16_fused_ce (fused_ce patch set) and switches it on with model.fused_ce.
    The v1-based candidate has neither, so this returns None there and the reference CE below is used."""
    fn = globals().get("rpv16_fused_ce")
    return fn if (callable(fn) and bool(getattr(model, "fused_ce", False))) else None


def ah_ce_mean(model, hh, yy):
    """Mean RPV CE and all per-row NLL. Fused NLL is detached, matching its SAT regret-target use."""
    if RPV16_TIED_HEAD:
        return tied_ce(model, hh, yy, return_token_nll=True)
    fn = ah_fused_ce_fn(model)
    if fn is not None:
        hint = float(getattr(model, "fused_ce_grad_scale_hint", getattr(model, "ce_grad_scale", 1.0)))
        return fn(hh, yy, model.factor_head.weight, model.mix_head.weight, ce_chunk=int(getattr(model, "ce_chunk", 4096)), grad_scale_hint=hint, return_token_nll=True)
    nll = ah_rpv_token_nll(model, hh, yy)
    return nll.mean(), nll


class AllHeadsRuntime:
    """All non-AR training state. Absent (None) under --objectives ar, which leaves v1 behaviour untouched."""
    def __init__(self, args, model, head_params, device):
        self.args = args
        self.seed = int(args.seed)
        self.device = device
        self.head_params = list(head_params)
        self.nat_attn = str(args.nat_attn)
        self.head_grad = str(args.allheads_head_grad)
        self.aux_lr_mult = float(args.allheads_aux_lr_mult)
        self.aux = AllHeadsAux(device)
        self.aux_opt = torch.optim.AdamW(list(self.aux.parameters()), lr=float(args.lr), betas=(0.9, 0.95), weight_decay=0.0)
        self.hot_path = Path(args.allheads_hot_config) if args.allheads_hot_config else Path(args.save_dir) / "allheads_hot.json"
        self._hot_mtime = None
        self.cli_cfg = dict(ALLHEADS_HOT_DEFAULTS)
        self.cli_cfg.update({
            "dblock_ar_prob": args.dblock_ar_prob, "dblock_sat_prob": args.dblock_sat_prob, "dblock_nat_prob": args.dblock_nat_prob,
            "sat_fixed_share": args.sat_fixed_share, "satvar_block_probs": args.satvar_block_probs,
            "satvar_lambda": args.satvar_lambda, "satvar_merge_cost": (None if args.satvar_merge_cost < 0 else args.satvar_merge_cost),
            "dblock_satvar_regret_weight": args.satvar_regret_weight, "satvar_log_every": args.satvar_log_every,
            "satvar_regret_max_blocks": getattr(args, "satvar_regret_max_blocks", 4096),
            "nat_mask_ratio": args.nat_mask_rate, "nat_span_mask_prob": args.nat_span_mask_prob,
            "nat_suffix_mask_prob": args.nat_suffix_mask_prob, "nat_region_partial_prob": args.nat_region_partial_prob})
        self.cfg = dict(self.cli_cfg)
        self._clock = None
        self._objective = "ar"
        self._fixed2 = False
        self.probs = ah_floor_probs(self._raw_probs())
        self.counts = {n: 0 for n in ALLHEADS_OBJECTIVES}
        self.sat_steps = 0
        self.info = {}
        self.step_info = {}
        self.debug = False
        self.debug_out = {}
        # AR-protective norm matching (make_v3_4_normmatch): per-stage log-EMA of the pre-clip AR grad norm
        self.nonar_clip_mult = float(getattr(args, "allheads_nonar_clip_mult", 1.0))
        self.nonar_clip_default = 0.02
        self.ar_gn_logema = {}
        self.ar_gn_n = {}
        self._last_nonar_max_norm = None
        n_aux = sum(p.numel() for p in self.aux.parameters())
        print(json.dumps({"event": "allheads_ready", "schema": ALLHEADS_SCHEMA, "objectives": list(ALLHEADS_OBJECTIVES),
                          "probs": self.probs, "prob_floor": ALLHEADS_PROB_FLOOR, "hot_config": str(self.hot_path),
                          "nat_attn": self.nat_attn, "nat_mask_id": NAT_MASK_ID, "head_grad": self.head_grad,
                          "aux_parameters_outside_core": n_aux,
                          "core_contract": "no new BF16/FP32 GEMM in the trainable transformer core; outside-core additions: shift_emb[2,256] fp32 add on the 256-d head input, SAT-var regret MLP 256-256-2 fp32 on a DETACHED input; SAT adds one extra flash-attn call per stage",
                          "satvar_contract": "always enabled; block sizes {1,2}; policy = learned CE regret + explicit speed reward lam; no disable switch, no forced constant stride"}), flush=True)

    # ---- configuration
    def _raw_probs(self):
        return {"ar": self.cfg["dblock_ar_prob"], "sat": self.cfg["dblock_sat_prob"], "nat": self.cfg["dblock_nat_prob"]}

    def reload_hot(self):
        try:
            mt = self.hot_path.stat().st_mtime_ns
        except OSError:
            mt = None
        if mt == self._hot_mtime:
            return False
        self._hot_mtime = mt
        cfg = dict(self.cli_cfg)
        applied = {}
        if mt is not None:
            try:
                obj = json.loads(self.hot_path.read_text())
                if not isinstance(obj, dict):
                    raise ValueError("hot config must be a JSON object")
                if "satvar_regret_weight" in obj and "dblock_satvar_regret_weight" not in obj:
                    obj["dblock_satvar_regret_weight"] = obj["satvar_regret_weight"]
                for k in ALLHEADS_HOT_DEFAULTS:
                    if k in obj:
                        cfg[k] = obj[k]; applied[k] = obj[k]
            except Exception as e:
                print(json.dumps({"event": "allheads_hot_config_rejected", "path": str(self.hot_path), "error": type(e).__name__, "detail": str(e)[:200], "kept": "previous values"}), flush=True)
                return False
        self.cfg = cfg
        # V89E_CLI_PROBS_WIN_RELOAD: argv --dblock-*-prob always win over hot/ckpt so AR densify binds after resume.
        self.cfg["dblock_ar_prob"] = float(self.args.dblock_ar_prob)
        self.cfg["dblock_sat_prob"] = float(self.args.dblock_sat_prob)
        self.cfg["dblock_nat_prob"] = float(self.args.dblock_nat_prob)
        self.cli_cfg["dblock_ar_prob"] = self.cfg["dblock_ar_prob"]
        self.cli_cfg["dblock_sat_prob"] = self.cfg["dblock_sat_prob"]
        self.cli_cfg["dblock_nat_prob"] = self.cfg["dblock_nat_prob"]
        self.probs = ah_floor_probs(self._raw_probs())
        print(json.dumps({"event": "allheads_hot_config", "path": str(self.hot_path), "present": mt is not None, "applied": applied, "probs": self.probs, "effective": self.effective()}), flush=True)
        return True

    def effective(self):
        c = self.cfg
        def _f(key, default, lo, hi):
            try:
                v = float(c.get(key, default))
                if v != v: v = default
            except Exception:
                v = default
            return min(hi, max(lo, v))
        lam = _f("satvar_lambda", 1.0, 1.0e-3, 100.0)              # explicit speed reward, never zero
        mc = c.get("satvar_merge_cost", None)
        try:
            mu = 0.5 * lam if mc is None else max(0.0, float(mc))
        except Exception:
            mu = 0.5 * lam
        return {"lam": lam, "mu": mu,
                "sat_fixed_share": _f("sat_fixed_share", 0.5, 0.0, 0.9),   # variable partitions always keep >= 10% of SAT updates
                "p2": ah_parse_block_probs(c.get("satvar_block_probs", "1:0.5,2:0.5")),
                "regret_weight": _f("dblock_satvar_regret_weight", 0.05, 1.0e-4, 1.0),
                "log_every": int(_f("satvar_log_every", 20, 1, 1_000_000)),
                "regret_max_blocks": int(_f("satvar_regret_max_blocks", 4096, 256, 1 << 20))}

    # ---- objective schedule
    def objective_for(self, clock, micro_in_update=0):
        clock = int(clock)
        if clock != self._clock:
            self.reload_hot()
            self._clock = clock
            self._objective = ah_draw_objective(self.seed, clock, self.probs)
            self._fixed2 = ah_u01(self.seed, clock, 1) < self.effective()["sat_fixed_share"]
        return self._objective

    def wants_head_grad(self, objective):
        return (objective == "sat" and self.head_grad in ("sat", "sat+nat")) or (objective == "nat" and self.head_grad == "sat+nat")

    # ---- forward
    def forward(self, objective, model, ids, labels, train_stage, clock, micro):
        if train_stage is None:
            raise RuntimeError("non-AR objectives run only on rotating stage updates; the head anchor stays AR")
        if (objective == "sat" or self.nat_attn == "bidir") and ah_routing_alignment(*ids.shape) == "misaligned":
            raise RuntimeError(f"batch x seq = {tuple(ids.shape)} couples DIFFERENT positions through the balanced router; non-causal objectives need batch*seq/{EXPERTS} to be a multiple of seq (production 24x2048 is) or a per-token-routed shape")
        if self.wants_head_grad(objective):
            for p in self.head_params:
                p.requires_grad_(True)       # model.set_trainable_stage() resets this at the next update
        if objective == "sat":
            return self._forward_sat(model, ids, labels, train_stage, clock, micro)
        if objective == "nat":
            return self._forward_nat(model, ids, labels, train_stage, clock, micro)
        raise ValueError(f"unknown objective {objective!r}")

    def _forward_sat(self, model, ids, labels, ts, clock, micro):
        eff = self.effective()
        B, T = ids.shape
        dev = ids.device
        gen = ah_generator(self.seed, clock, 2, micro)
        # ONE partition shared by every row (as S does). With per-row partitions the router's cross-row coupling at
        # equal positions opens a path  token p -> row r' (pairs p-1,p) -> routing group p-1 -> row r position p-1,
        # whose target can be token p. A shared partition closes it: every position that can be influenced by token p
        # has block_end >= p in EVERY row, and targets always lie beyond block_end.
        first2, slot, bsize, bend = (x.expand(B, T) for x in ah_sat_partition(1, T, eff["p2"], self._fixed2, gen))
        first2_d = first2.to(dev)
        with ah_attn_mode("sat", first2_d):
            x, aux_total, route_counts = ah_trunk(model, ids, ts)
        h = model.out_proj(model.norm(x))
        slot_d = slot.to(dev).reshape(-1); bsize_d = bsize.to(dev)
        # slot j of a block predicts the token at pos + blocksize (first token the block cannot see).
        # labels[:, i] is token i+1, so token pos+bsize is labels[:, pos+bsize-1].
        tgt_idx = torch.arange(T, device=dev).view(1, T) + bsize_d - 1
        valid = (tgt_idx <= T - 1).reshape(-1)
        sel = valid.nonzero(as_tuple=False).flatten()
        hflat = h.reshape(-1, TOKEN_DIM)
        hh = self.aux.slot_hidden(hflat.index_select(0, sel), slot_d.index_select(0, sel))
        y_all = labels.gather(1, tgt_idx.clamp(max=T - 1)).reshape(-1)
        yy = y_all.index_select(0, sel)
        ce, nll = ah_ce_mean(model, hh, yy)
        # ---- SAT-var regret head: regress log per-slot CE from the block's LAST hidden state (detached)
        nll_grid = torch.zeros(B * T, device=dev, dtype=torch.float32)
        second2_d = (slot_d == 1)
        last = ((bsize_d.reshape(-1) == 1) | second2_d) & valid          # every slot of the block has a target
        last_idx = last.nonzero(as_tuple=False).flatten()
        if nll is not None:
            nll_grid.index_copy_(0, sel, nll.detach().float())
        else:
            # Compatibility fallback for external CE adapters without row NLL. The embedded full-NLL kernel
            # never takes this path. Like S (which samples 256 blocks), take regret targets from an
            # exact no-grad reference CE on a sampled subset of blocks, drawn from the same dedicated generator.
            cap = int(eff["regret_max_blocks"])
            if int(last_idx.numel()) > cap:
                pick = torch.randperm(int(last_idx.numel()), generator=gen)[:cap].sort().values.to(dev)
                last_idx = last_idx.index_select(0, pick)
            _is2 = second2_d.index_select(0, last_idx)
            rows = torch.cat([last_idx, last_idx[_is2] - 1])
            with torch.no_grad():
                hr = self.aux.slot_hidden(hflat.index_select(0, rows), slot_d.index_select(0, rows)).detach()
                nll_grid.index_copy_(0, rows, ah_rpv_token_nll(model, hr, y_all.index_select(0, rows)).float())
        pred = self.aux.regret_log_ce(hflat.index_select(0, last_idx))   # [M,2]
        is2 = second2_d.index_select(0, last_idx)
        idx2 = last_idx[is2]
        idx1 = last_idx[~is2]
        tlog = lambda v: torch.log(v.clamp_min(1.0e-3))
        pred_parts = [pred[~is2, 0], pred[is2, 0], pred[is2, 1]]
        tgt_parts = [tlog(nll_grid.index_select(0, idx1)), tlog(nll_grid.index_select(0, idx2 - 1)), tlog(nll_grid.index_select(0, idx2))]
        pred_sel = torch.cat(pred_parts); target_log = torch.cat(tgt_parts)
        regret = F.smooth_l1_loss(pred_sel, target_log, beta=0.5) if pred_sel.numel() else torch.zeros((), device=dev)
        loss = ce + 0.01 * aux_total / STAGES + eff["regret_weight"] * regret
        n1 = int((bsize == 1).sum()); n2 = int(first2.sum())
        self.sat_steps += 1
        info = {"objective": "sat", "n_targets": int(sel.numel()), "sat_ce": float(ce.detach()), "sat_regret_loss": float(regret.detach()),
                "ce_path": "fused_full_nll" if ah_fused_ce_fn(model) is not None else "reference", "regret_blocks": int(last_idx.numel()),
                "sat": {"fixed": bool(self._fixed2), "blocks1": n1, "blocks2": n2, "mean_block": round((n1 + 2 * n2) / max(1, n1 + n2), 4),
                        "frac_tokens_in_2": round(2 * n2 / max(1, B * T), 4), "p2": eff["p2"], "lam": eff["lam"], "regret_weight": eff["regret_weight"]}}
        if self.sat_steps % eff["log_every"] == 0 or self.sat_steps == 1:
            with torch.no_grad():
                if int(idx2.numel()) > 0:
                    gate = ah_satvar_gate_stats(nll_grid.index_select(0, idx2 - 1), nll_grid.index_select(0, idx2), pred[is2].detach(), eff["lam"], eff["mu"])
                else:
                    gate = {"n_blocks2": 0}
                # Symbolic baseline for the learned cost: the best CONSTANT per slot (this batch's own mean log-CE, i.e. an
                # oracle constant, so the comparison is conservative).  Predictions are made BEFORE this batch is trained
                # on, so regret_mae_log is an online held-out error; skill > 0 means the head beats the constant.
                mae = mae_const = skill = None
                if pred_sel.numel():
                    n_a, n_b = int(idx1.numel()), int(idx2.numel())
                    col0 = target_log[: n_a + n_b]; col1 = target_log[n_a + n_b:]
                    dev_abs = torch.cat([(c - c.mean()).abs() for c in (col0, col1) if c.numel()])
                    mae = float((pred_sel - target_log).abs().mean()); mae_const = float(dev_abs.mean())
                    skill = (1.0 - mae / mae_const) if mae_const > 0.0 else None
                gate.update({"event": "satvar_gate", "sat_step": self.sat_steps, "optimizer_update_clock": int(clock), "train_stage": int(ts),
                             "partition": "fixed2" if self._fixed2 else "variable", "regret_updates": int(self.aux.regret_updates),
                             "regret_mae_log": None if mae is None else round(mae, 4),
                             "regret_mae_log_const_baseline": None if mae_const is None else round(mae_const, 4),
                             "regret_skill_vs_const": None if skill is None else round(skill, 4)})
            print(json.dumps(gate), flush=True)
            info["satvar_gate"] = {k: gate.get(k) for k in ("base_rate", "pred_rate", "precision", "recall", "lift", "mean_stride_policy", "regret_skill_vs_const")}
        if self.debug:
            self.debug_out = {"h": h.detach(), "sel": sel, "yy": yy, "first2": first2, "slot": slot, "bsize": bsize, "bend": bend, "nll": (None if nll is None else nll.detach()), "tgt_idx": tgt_idx, "nll_grid": nll_grid, "last_idx": last_idx}
        self.info = info
        return loss, aux_total / STAGES, route_counts

    def _forward_nat(self, model, ids, labels, ts, clock, micro):
        gen = ah_generator(self.seed, clock, 3, micro)
        model_ids, mask, stats = ah_nat_corrupt(ids, self.cfg, gen)
        if self.nat_attn == "bidir":
            with ah_attn_mode("nat_bidir"):
                x, aux_total, route_counts = ah_trunk(model, model_ids, ts)
        else:
            # causal denoiser: identical attention kernel/windows to AR. The network has no positional
            # encoding, so causality is its only position signal (see allheads_ready / self-tests).
            x, aux_total, route_counts = ah_trunk(model, model_ids, ts)
        h = model.out_proj(model.norm(x))
        # true denoising target: ORIGINAL token i at masked position i; rows gathered BEFORE the
        # 256->12288 projection (natmaskedhead_v2 head compaction).
        sel = mask.reshape(-1).nonzero(as_tuple=False).flatten()
        hh = h.reshape(-1, TOKEN_DIM).index_select(0, sel)
        yy = ids.reshape(-1).index_select(0, sel)
        ce, nll = ah_ce_mean(model, hh, yy)
        loss = ce + 0.01 * aux_total / STAGES
        self.info = {"objective": "nat", "n_targets": int(sel.numel()), "nat_ce": float(ce.detach()), "ce_path": "fused_full_nll" if ah_fused_ce_fn(model) is not None else "reference", "nat": dict(stats, attn=self.nat_attn)}
        if self.debug:
            self.debug_out = {"h": h.detach(), "sel": sel, "yy": yy, "mask": mask, "model_ids": model_ids}
        return loss, aux_total / STAGES, route_counts

    # ---- AR-protective gradient norm matching
    def note_ar_grad_norm(self, stage, grad_norm):
        """Record the PRE-clip grad norm of an AR stage update (never called for the head anchor or non-AR updates)."""
        if stage is None:
            return
        g = float(grad_norm)
        if not (g == g) or g <= 0.0 or g == float("inf"):
            return
        s = int(stage)
        lg = math.log(g)
        n = self.ar_gn_n.get(s, 0)
        self.ar_gn_logema[s] = lg if n == 0 else 0.9 * self.ar_gn_logema[s] + 0.1 * lg
        self.ar_gn_n[s] = n + 1

    def nonar_max_norm(self, stage):
        """Clip threshold for a SAT/NAT stage update: mult x the stage's typical AR grad norm, never above v1's 1.0."""
        if self.nonar_clip_mult <= 0.0:
            self._last_nonar_max_norm = 1.0
            return 1.0
        s = None if stage is None else int(stage)
        ref = math.exp(self.ar_gn_logema[s]) if self.ar_gn_n.get(s, 0) >= 4 else self.nonar_clip_default
        m = min(1.0, max(1.0e-4, self.nonar_clip_mult * ref))
        self._last_nonar_max_norm = m
        return m

    # ---- optimizer side
    def after_step(self, objective, lr_now, head_opt):
        self.step_info = {}
        self.counts[objective] = self.counts.get(objective, 0) + 1
        if objective == "ar":
            return
        ps = [p for p in self.aux.parameters() if p.grad is not None]
        if ps:
            gn = torch.nn.utils.clip_grad_norm_(ps, 1.0)
            if not torch.isfinite(gn): raise RuntimeError("nonfinite allheads aux grad norm")
            for pg in self.aux_opt.param_groups: pg["lr"] = float(lr_now) * self.aux_lr_mult
            self.aux_opt.step()
            self.aux.regret_updates += 1
            self.step_info["aux_grad_norm"] = float(gn)
        self.aux_opt.zero_grad(set_to_none=True)
        if self.wants_head_grad(objective):
            gh = torch.nn.utils.clip_grad_norm_(self.head_params, 1.0)
            if not torch.isfinite(gh): raise RuntimeError("nonfinite head grad norm on a non-AR update")
            for pg in head_opt.param_groups: pg["lr"] = float(lr_now)
            head_opt.step()
            head_opt.zero_grad(set_to_none=True)
            self.step_info["head_grad_norm"] = float(gh)

    def telemetry(self, objective, n_tokens):
        if objective == "ar":
            out = {"objective": "ar", "n_targets": int(n_tokens)}
        else:
            out = dict(self.info)
            out["nonar_max_norm"] = self._last_nonar_max_norm
        out.update(self.step_info)
        out["objective_probs"] = self.probs
        out["objective_counts"] = dict(self.counts)
        self.step_info = {}
        return out

    # ---- checkpoint (schema v6 = v5 + this blob)
    def ckpt_extra(self):
        return {"allheads": {"schema": ALLHEADS_SCHEMA, "aux": self.aux.state_dict(), "aux_optimizer": self.aux_opt.state_dict(),
                             "objective_counts": dict(self.counts), "sat_steps": int(self.sat_steps),
                             "config": {"hot": {k: self.cfg.get(k) for k in ALLHEADS_HOT_DEFAULTS}, "effective": self.effective(), "probs": self.probs,
                                        "nat_attn": self.nat_attn, "nat_mask_id": NAT_MASK_ID, "head_grad": self.head_grad,
                                        "satvar_kmax": SATVAR_KMAX, "shift_on": "head_input_256", "regret_input": "head_input_256_detached_block_last",
                                        "sat_attention": "block-causal {1,2}: call A causal(window w) + call B causal(window w+1) over K/V right-padded by one; first-of-pair rows take call B",
                                        "sat_target": "slot j at pos predicts token pos+blocksize", "sat_partition": "one {1,2} partition per micro-batch, shared by all rows"}}}

    def load_from_ckpt(self, ck):
        """Only a genuinely pre-SAT v5 checkpoint may initialize a new policy.
    
        An existing allheads checkpoint must restore its learned weights, optimizer,
        counters and effective policy configuration. Corruption is never interpreted
        as permission to silently reset the gate or discard its training state.
        """
        if not isinstance(ck, dict):
            raise RuntimeError("allheads resume requires a checkpoint dictionary")
        blob = ck.get("allheads")
        if blob is None:
            if ck.get("schema") != "agillm.gb10.1pf.recovery.v5":
                raise RuntimeError("only legacy v5 may initialize a missing allheads policy")
            print(json.dumps({"event": "allheads_fresh_init", "from_schema": ck["schema"],
                "to_schema": ALLHEADS_CKPT_SCHEMA, "fresh_modules": [
                    "allheads.aux.shift_emb (zeros)", "allheads.aux.stride_regret (prior-CE bias)",
                    "allheads.aux_optimizer (empty AdamW)"],
                "reason": "explicit v5-to-allheads migration; existing core state is not reset"}), flush=True)
            return "fresh"
        if not isinstance(blob, dict) or blob.get("schema") != ALLHEADS_SCHEMA:
            raise RuntimeError("allheads checkpoint schema mismatch")
        for key in ("aux", "aux_optimizer", "objective_counts", "config"):
            if not isinstance(blob.get(key), dict):
                raise RuntimeError("allheads checkpoint missing or malformed " + key)
        cfg = blob["config"]
        if cfg.get("satvar_kmax") != 2 or cfg.get("nat_mask_id") != NAT_MASK_ID:
            raise RuntimeError("allheads checkpoint stride or NAT mask identity changed")
        if cfg.get("nat_attn") not in ("causal", "bidir"):
            raise RuntimeError("allheads checkpoint invalid NAT attention mode")
        if cfg.get("head_grad") not in ("none", "sat", "sat+nat"):
            raise RuntimeError("allheads checkpoint invalid head gradient ownership")
        if not isinstance(cfg.get("hot"), dict):
            raise RuntimeError("allheads checkpoint lacks its saved objective configuration")
        unknown = set(cfg["hot"]) - set(ALLHEADS_HOT_DEFAULTS)
        if unknown:
            raise RuntimeError("allheads checkpoint has unsupported configuration keys: " + str(sorted(unknown)))
        counts = blob["objective_counts"]
        if set(counts) != set(ALLHEADS_OBJECTIVES) or any(type(v) is not int or v < 0 for v in counts.values()):
            raise RuntimeError("allheads checkpoint invalid objective counters")
        if type(blob.get("sat_steps")) is not int or blob["sat_steps"] < 0:
            raise RuntimeError("allheads checkpoint invalid SAT-step counter")
        def finite(x):
            if torch.is_tensor(x):
                return bool(torch.isfinite(x).all())
            if isinstance(x, dict):
                return all(finite(v) for v in x.values())
            if isinstance(x, (list, tuple)):
                return all(finite(v) for v in x)
            return not isinstance(x, float) or math.isfinite(x)
        if not finite(blob):
            raise RuntimeError("allheads checkpoint contains nonfinite learned state")
        # AdamW accepts some malformed state tensors at load time and fails only
        # during the next update. Validate parameter identity and geometry first.
        saved_aux = blob["aux"]
        live_aux = self.aux.state_dict()
        if set(saved_aux) != set(live_aux):
            raise RuntimeError("allheads auxiliary state keys mismatch")
        for key, target in live_aux.items():
            value = saved_aux[key]
            if not torch.is_tensor(value) or value.shape != target.shape or value.dtype != target.dtype:
                raise RuntimeError("allheads auxiliary tensor shape/dtype mismatch: " + key)
        updates = saved_aux["regret_updates"]
        if updates.numel() != 1 or int(updates) < 0:
            raise RuntimeError("allheads invalid learned-policy update counter")
        opt = blob["aux_optimizer"]
        saved_groups, saved_state = opt.get("param_groups"), opt.get("state")
        if not isinstance(saved_groups, list) or not isinstance(saved_state, dict):
            raise RuntimeError("allheads optimizer state/groups missing")
        if len(saved_groups) != len(self.aux_opt.param_groups):
            raise RuntimeError("allheads optimizer group count mismatch")
        mapped = {}
        for saved, current in zip(saved_groups, self.aux_opt.param_groups):
            ids = saved.get("params") if isinstance(saved, dict) else None
            if not isinstance(ids, list) or len(ids) != len(current["params"]):
                raise RuntimeError("allheads optimizer parameter count mismatch")
            for pid, param in zip(ids, current["params"]):
                if type(pid) is not int or pid in mapped:
                    raise RuntimeError("allheads optimizer duplicate/invalid parameter identity")
                mapped[pid] = param
            betas = saved.get("betas")
            if not isinstance(betas, (list, tuple)) or len(betas) != 2 or not all(0 <= float(b) < 1 for b in betas):
                raise RuntimeError("allheads optimizer invalid betas")
            if float(saved.get("lr", -1)) < 0 or float(saved.get("eps", 0)) <= 0 or float(saved.get("weight_decay", -1)) < 0:
                raise RuntimeError("allheads optimizer invalid learning hyperparameters")
        if any(pid not in mapped for pid in saved_state):
            raise RuntimeError("allheads optimizer state has unknown parameter identity")
        if int(updates) > 0 and not saved_state:
            raise RuntimeError("trained allheads policy has no optimizer moments")
        for pid, state in saved_state.items():
            if not isinstance(state, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
                raise RuntimeError("allheads optimizer incomplete Adam state")
            step = state["step"]
            if torch.is_tensor(step):
                if step.numel() != 1: raise RuntimeError("allheads optimizer step is not scalar")
                step = float(step)
            if not isinstance(step, (int, float)) or step < 0 or step != int(step):
                raise RuntimeError("allheads optimizer step is negative or fractional")
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key not in state: continue
                value = state[key]
                if not torch.is_tensor(value) or value.shape != mapped[pid].shape or not value.is_floating_point():
                    raise RuntimeError("allheads optimizer moment geometry mismatch: " + key)
        # Rejection must preserve the already installed policy, including when a
        # later saved-configuration check fails after state_dict loading succeeds.
        import copy as _ah_copy
        _aux_before = _ah_copy.deepcopy(self.aux.state_dict())
        _opt_before = _ah_copy.deepcopy(self.aux_opt.state_dict())
        _fields = ("counts", "sat_steps", "cfg", "cli_cfg", "nat_attn", "head_grad", "probs", "_clock", "_hot_mtime")
        _before = {key: _ah_copy.deepcopy(getattr(self, key)) for key in _fields}
        try:
            self.aux.load_state_dict(blob["aux"], strict=True)
            self.aux_opt.load_state_dict(blob["aux_optimizer"])
            self.counts = dict(counts)
            self.sat_steps = blob["sat_steps"]
            # Saved policy is the restart baseline. A subsequently read hot-config file
            # remains the explicit mechanism to change it after restoration.
            restored_cfg = dict(self.cli_cfg)
            restored_cfg.update(cfg["hot"])
            # V89_CLI_PROBS_WIN: CLI --dblock-*-prob beat ckpt hot so AR densify (e.g. 0.985 under floor 0.005) can bind after resume.
            restored_cfg["dblock_ar_prob"] = float(self.args.dblock_ar_prob)
            restored_cfg["dblock_sat_prob"] = float(self.args.dblock_sat_prob)
            restored_cfg["dblock_nat_prob"] = float(self.args.dblock_nat_prob)
            self.cfg = restored_cfg
            self.cli_cfg = dict(restored_cfg)
            self.nat_attn = cfg["nat_attn"]
            self.head_grad = cfg["head_grad"]
            self.probs = ah_floor_probs(self._raw_probs())
            effective = self.effective()
            old_effective = cfg.get("effective")
            if isinstance(old_effective, dict) and effective != old_effective:
                raise RuntimeError("saved allheads policy cannot be restored exactly")
            self._clock = None
            self._hot_mtime = None
            print(json.dumps({"event": "allheads_resumed", "schema": blob["schema"],
                "aux_optimizer": "resumed", "policy_config": "restored_from_checkpoint",
                "regret_updates": int(self.aux.regret_updates), "objective_counts": dict(self.counts),
                "effective": effective}), flush=True)
            return "resumed"
        except Exception:
            self.aux.load_state_dict(_aux_before, strict=True)
            self.aux_opt.load_state_dict(_opt_before)
            for key, value in _before.items(): setattr(self, key, value)
            raise


def allheads_setup(args, model, head_params, device="cuda"):
    spec = sorted(x.strip().lower() for x in str(args.objectives).split(",") if x.strip())
    if spec == ["ar"]:
        print(json.dumps({"event": "allheads_disabled_operator_flag", "objectives": "ar", "note": "v1 AR-only behaviour reproduced exactly; this violates the all-objectives owner contract and is meant for A/B and bisecting only"}), flush=True)
        return None
    if spec != sorted(ALLHEADS_OBJECTIVES):
        raise SystemExit("--objectives must be 'ar,sat,nat' (owner contract: all objectives stay active) or 'ar' (exact v1 reproduction)")
    return AllHeadsRuntime(args, model, head_params, device)
# <<< ALLHEADS END


def build_model():
    # V91P_SM120 2026-09-27: the architecture stays locked to the Blackwell SM12x block-scaled sparse NVFP4 MMA
    # family. GB10 (sm_121) and RTX 50xx / RTX PRO Blackwell (sm_120) execute the identical mxf4nvf4 4:8 sparse
    # path; the runtime libraries are rebuilt per ISA (sm_120a x86_64 here). Non-SM12x GPUs are still refused.
    if not torch.cuda.is_available():
        raise RuntimeError("AGILLM-GB10-1PF requires a CUDA GPU")
    cap = torch.cuda.get_device_capability(0)
    if not sparse_backend_emulated() and cap[0] != 12:
        raise RuntimeError("native backend requires a Blackwell SM12x GPU; set AGILLM_SPARSE_BACKEND=auto|emulate")
    if cap < (8, 0):
        raise RuntimeError("AGILLM-GB10-1PF needs BF16 tensor cores (sm_80+); got sm_%d%d" % cap)
    print(json.dumps(sparse_backend_info()), flush=True)   # V91P3_PORTABLE
    return Model()


def profile():
    out={
        "name":NAME,"architecture":"hardware_locked_v2","parameters":parameter_accounting(),
        "training_tokens":TOTAL_TRAIN_TOKENS,"tokens_per_parameter":TOTAL_TRAIN_TOKENS/parameter_accounting()["total"],
        "vocab":VOCAB,"token_dim":TOKEN_DIM,"model_dim":D,"stages":STAGES,"experts_per_stage":EXPERTS,
        "expert_shape":[D,FFN],"attention":{"q_heads":Q_HEADS,"kv_heads":KV_HEADS,"head_dim":HEAD_DIM,"windows":WINDOWS},
        "sparse_target_tflops":1000.0,"verified_sparse_tflops":VERIFIED_SPARSE_TFLOPS,
        "tokenizer":str(TOKENIZER_JSON),"hf_sources":HF_SOURCES,
    }
    if torch.cuda.is_available(): out["device"]={"name":torch.cuda.get_device_name(0),"memory":torch.cuda.get_device_properties(0).total_memory}
    print(json.dumps(out,indent=2))


def dataset_probe(names):
    s=HFStream(names)
    src,text=s.next_text()
    ids=s.tok.encode(text).ids
    print(json.dumps({"ok":True,"source":src,"chars":len(text),"tokens":len(ids),"head_ids":ids[:16]},indent=2))


# ---- v2 full-stack host helpers ---------------------------------------------------------
_STOP = {"n": 0, "sig": None}
_BACKGROUND = {"stream": None, "writer": None, "ckpt_state": None, "flush": None}

def _on_signal(signum, frame):
    # Flag only. The loop finishes the current update, saves, and exits 0. Repeated signals
    # just count, so a second SIGTERM/SIGINT can never interrupt the save it triggered.
    _STOP["n"] += 1; _STOP["sig"] = int(signum)


def _mem_available_bytes():
    """Host bytes that can be taken without swapping: min(MemAvailable, cgroup-v2 headroom). None if unknown."""
    avail = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
                break
    except Exception:
        pass
    try:
        mx = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if mx != "max":
            head = int(mx) - int(Path("/sys/fs/cgroup/memory.current").read_text())
            for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
                if line.startswith("inactive_file "):
                    head += int(line.split()[1])
                    break
            avail = head if avail is None else min(avail, head)
    except Exception:
        pass
    return avail


def _host_snapshot(obj, memo):
    """Deep host copy of a checkpoint payload for the background writer.

    Keeps container types and key order, OrderedDict attributes (state_dict _metadata), object
    identity sharing, storage sharing and full storage extents (torch.save writes whole
    storages) and tensor python attributes (bitsandbytes is_paged/page_deviceid), so
    torch.save(snapshot) pickles the same structure and tensor bytes as torch.save(live).
    Fails closed (TypeError) on anything it cannot mirror; the caller then saves synchronously.
    """
    if isinstance(obj, torch.Tensor):
        if type(obj) is not torch.Tensor or obj.layout is not torch.strided:
            raise TypeError(f"cannot host-snapshot tensor type {type(obj).__name__}/{obj.layout}")
        st = obj.untyped_storage(); nbytes = st.nbytes(); key = ("storage", st.data_ptr(), nbytes)
        host = memo.get(key)
        if host is None:
            host = torch.empty(nbytes, dtype=torch.uint8)
            if nbytes:
                host.copy_(torch.empty(0, dtype=torch.uint8, device=obj.device).set_(st))
                memo[key] = host
        out = torch.empty(0, dtype=obj.dtype).set_(host.untyped_storage(), obj.storage_offset(), obj.size(), obj.stride())
        if obj.requires_grad: out.requires_grad_(True)
        if obj.__dict__: out.__dict__.update(obj.__dict__)
        return out
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj
    key = ("id", id(obj))
    if key in memo:
        return memo[key][1]
    if isinstance(obj, dict):
        out = obj.__class__()
        memo[key] = (obj, out)
        for k, v in obj.items(): out[k] = _host_snapshot(v, memo)
        if getattr(obj, "__dict__", None): out.__dict__.update(copy.deepcopy(obj.__dict__))
        return out
    if isinstance(obj, list):
        out = []
        memo[key] = (obj, out)
        out.extend(_host_snapshot(v, memo) for v in obj)
        return out
    if isinstance(obj, tuple):
        items = [_host_snapshot(v, memo) for v in obj]
        if all(a is b for a, b in zip(items, obj)):
            out = obj  # immutable all the way down (rng state, betas): share it
        else:
            out = tuple(items) if type(obj) is tuple else obj.__class__(*items)
        memo[key] = (obj, out)
        return out
    out = copy.deepcopy(obj)
    memo[key] = (obj, out)
    return out


def _fsync_best_effort(path):
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    except OSError:
        pass


# ---- v84 checkpoint save guard ------------------------------------------------------------
# Contract: a checkpoint emission failure (OSError incl. disk full/EIO/read-only, RuntimeError
# from torch serialization, pickle errors) is logged with a traceback, its partial tmp is
# deleted, and training continues; the save is retried at the next save interval (or at
# signal-exit). CUDA faults and NaN guards are NOT caught (they still stop the run); the
# training step itself is never wrapped. On disk-full (precheck or ENOSPC) the guard frees
# space ONCE per save under the shared prune lock -- (a) this trainer's own leftover .tmp
# partials, then (b) receipted + Hub-verified freezes/old checkpoints in the save dir, oldest
# first -- and retries the save once. Nothing safe to delete => skip the save, keep training.
# After _CKPT_GUARD_LOUD_AFTER consecutive failures it logs LOUDLY but still does not die.
import traceback, pickle, errno as _ck_errno, fcntl as _ck_fcntl, re as _ck_re
_CKPT_GUARD_LOUD_AFTER = 3
_CKPT_GUARD_MIN_BYTES = int(os.environ.get("AGILLM_CKPT_GUARD_MIN_BYTES", "1000000"))
_CKPT_VERIFY_RELOAD = os.environ.get("AGILLM_CKPT_VERIFY_RELOAD", "1") != "0"
_CKPT_CATCH = (OSError, RuntimeError, pickle.PickleError, TypeError, EOFError)  # TypeError = "cannot pickle ..."
_PRUNE_LOCK_PATH = os.environ.get("AGILLM_PRUNE_LOCK", "/workspace/.agillm_prune.lock")
_PRUNE_LOG_PATH = os.environ.get("AGILLM_PRUNE_LOG", "/workspace/.agillm_prune_log.jsonl")
_PRUNE_LOCK_TIMEOUT_S = float(os.environ.get("AGILLM_PRUNE_LOCK_TIMEOUT_S", "30"))
_PRUNE_HUB_TIMEOUT_S = float(os.environ.get("AGILLM_PRUNE_HUB_TIMEOUT_S", "20"))
_PRUNE_DEADLINE_S = float(os.environ.get("AGILLM_PRUNE_DEADLINE_S", "120"))
_PRUNE_REPO = os.environ.get("AGILLM_PRUNE_REPO", "MarxistLeninist/AGILLM-GB10-1PF")
_PRUNE_LANE = "rpv16"


def _atomic_write_text(path, text):
    path = Path(path); tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "w") as fh:
            fh.write(text); fh.flush()
            try: os.fsync(fh.fileno())
            except OSError: pass
        os.replace(tmp, path)
    except BaseException:
        try: tmp.unlink()
        except OSError: pass
        raise


def _is_cuda_fault(e):
    if "AcceleratorError" in type(e).__name__: return True
    if not isinstance(e, RuntimeError): return False
    msg = str(e)
    return any(k in msg for k in ("CUDA error", "CUDA out of memory", "cuda runtime error", "device-side assert",
                                  "illegal memory access", "CUBLAS_STATUS", "cuDNN error", "NCCL error"))


def _is_disk_full(e, need=None, path=None):
    if isinstance(e, OSError) and e.errno in (_ck_errno.ENOSPC, getattr(_ck_errno, "EDQUOT", -1)): return True
    msg = str(e)
    if "No space left" in msg or "ENOSPC" in msg or "Disk quota" in msg: return True
    # torch's zip writer surfaces a real ENOSPC as an errno-less iostream error; free_fn() decides and the retry is bounded to one
    if "basic_ios::clear" in msg or "iostream error" in msg: return True
    if ("write failed" in msg or "file write" in msg) and need and path is not None:
        try: return shutil.disk_usage(path).free < need
        except OSError: return False
    return False


def _ckpt_guard_write_status(ckpt_state, step=None):
    """Mirror for the disk keeper: <save_dir>/ckpt_save_status.json (atomic)."""
    p = (ckpt_state or {}).get("status_path")
    if not p: return
    try:
        try: _free = int(shutil.disk_usage(Path(p).parent).free)
        except OSError: _free = None
        _need = ckpt_state.get("last_need_bytes")
        _bn = max(0, int(_need) - _free) if (_need is not None and _free is not None) else None  # est. ckpt size (+1GiB margin) minus free
        _atomic_write_text(p, json.dumps({
            "lane": _PRUNE_LANE, "step": step,
            "consecutive_failures": ckpt_state.get("ckpt_save_failures_consecutive", 0),
            "total_failures": ckpt_state.get("ckpt_save_failures_total", 0),
            "last_error": ckpt_state.get("ckpt_last_error"),
            "last_reason": ckpt_state.get("ckpt_last_skip_reason"),
            "last_ok_step": ckpt_state.get("ckpt_last_success_step"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bytes_needed": _bn, "need_bytes": _need, "free_bytes": _free,
        }) + "\n")
    except Exception:
        pass


# >>> V43KN_HOSTMEM BEGIN (feell-rpv16-lane 2026-10-06; execution-only, no numerics)
_HM_ENABLED = os.environ.get("AGILLM_MALLOC_TRIM", "1") != "0"
_HM_TRIM_EVERY = int(os.environ.get("AGILLM_MALLOC_TRIM_EVERY_UPDATES", "256") or 0)
_HM_MIN_RSS = int(float(os.environ.get("AGILLM_MALLOC_TRIM_MIN_RSS_GB", "30") or 0) * (1 << 30))  # V43KO gate
_HM_STATS = {"trims": 0, "released_bytes": 0, "failures": 0}
_HM_LIBC = None


def _hm_rss_anon():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _hm_trim(reason, step=None):
    """gc.collect() + glibc malloc_trim(0): return free heap pages to the OS. Never raises."""
    global _HM_LIBC
    if not _HM_ENABLED:
        return None
    try:
        import ctypes, gc
        if _HM_LIBC is None:
            _HM_LIBC = ctypes.CDLL("libc.so.6")
            _HM_LIBC.malloc_trim.argtypes = [ctypes.c_size_t]
            _HM_LIBC.malloc_trim.restype = ctypes.c_int
        t0 = time.perf_counter(); before = _hm_rss_anon()
        if _HM_MIN_RSS > 0 and before is not None and before < _HM_MIN_RSS:
            return None  # V43KO: below RSS gate, skip trim (avoids page-fault churn / slower save snapshots)
        gc.collect()
        rc = _HM_LIBC.malloc_trim(0)
        after = _hm_rss_anon(); dt = time.perf_counter() - t0
        rel = (before - after) if (before is not None and after is not None) else None
        _HM_STATS["trims"] += 1; _HM_STATS["released_bytes"] += max(0, rel or 0)
        ev = {"event": "host_mem_trim", "schema": "feell.v43ko.hostmem.v1", "min_rss_gb": _HM_MIN_RSS / (1 << 30), "reason": reason, "step": step,
              "rss_anon_before": before, "rss_anon_after": after, "released_bytes": rel, "malloc_trim_rc": rc,
              "trim_s": round(dt, 4), "trims": _HM_STATS["trims"], "released_total": _HM_STATS["released_bytes"]}
        _emit(ev)
        return ev
    except Exception as e:
        _HM_STATS["failures"] += 1
        try:
            _emit({"event": "host_mem_trim_failed", "reason": reason, "error": f"{type(e).__name__}: {e}"[:200]})
        except Exception:
            pass
        return None
# <<< V43KN_HOSTMEM END


def _ckpt_guard_note_success(ckpt_state, step=None):
    if ckpt_state is None: return
    n = ckpt_state.get("ckpt_save_failures_consecutive", 0)
    ckpt_state["ckpt_save_failures_consecutive"] = 0
    ckpt_state["ckpt_last_success_step"] = step; ckpt_state["ckpt_last_success_ts"] = time.time()
    if n:
        try: _emit({"event": "checkpoint_guard_recovered", "ts": time.time(), "step": step, "after_consecutive_failures": n})
        except Exception: pass
    _ckpt_guard_write_status(ckpt_state, step)
    _hm_trim("checkpoint", step)  # V43KN: release the save's transient host buffers


def _ckpt_guard_note_failure(ckpt_state, step, reason, kind, error, tb=None):
    n = 1; total = 1
    if ckpt_state is not None:
        n = ckpt_state["ckpt_save_failures_consecutive"] = ckpt_state.get("ckpt_save_failures_consecutive", 0) + 1
        total = ckpt_state["ckpt_save_failures_total"] = ckpt_state.get("ckpt_save_failures_total", 0) + 1
        ckpt_state["ckpt_last_error"] = f"{kind}: {error}"[:500]; ckpt_state["ckpt_last_error_ts"] = time.time()
        ckpt_state["ckpt_last_error_step"] = step
        if kind not in ("skipped", "enospc_skip"): ckpt_state["ckpt_last_skip_reason"] = f"{kind}:{reason}"
    ev = {"event": "checkpoint_guard_failure", "ts": time.time(), "kind": kind, "step": step, "reason": reason,
          "error": str(error)[:400], "ckpt_save_failures_consecutive": n, "ckpt_save_failures_total": total,
          "action": "partial tmp removed; training continues; retry at next save interval"}
    if tb: ev["traceback"] = tb[-4000:]
    try: _emit(ev)
    except Exception: pass
    if n >= _CKPT_GUARD_LOUD_AFTER:
        try:
            _emit({"event": "checkpoint_guard_CRITICAL", "ts": time.time(), "ckpt_save_failures_consecutive": n, "step": step,
                   "last_error": str(error)[:400], "message": f"{n} consecutive checkpoint saves FAILED; trainer is still training but nothing new is being persisted. Fix disk/permissions."})
            sys.stderr.write(f"\n!!!!!!!! CHECKPOINT GUARD: {n} CONSECUTIVE SAVE FAILURES (step {step}): {str(error)[:300]} -- STILL TRAINING, NOT PERSISTING !!!!!!!!\n")
            sys.stderr.flush()
        except Exception: pass
    _ckpt_guard_write_status(ckpt_state, step)
    return n


def _hub_remote_info(repo, path_in_repo, timeout):
    """(info, err). info={'size','sha256'} of the Hub file; bounded by `timeout` (daemon thread)."""
    box = {}
    def run():
        try:
            from huggingface_hub import HfApi
            box["infos"] = list(HfApi().get_paths_info(repo, [path_in_repo], repo_type="model", expand=True))
        except BaseException as e:
            box["err"] = f"{type(e).__name__}: {e}"[:300]
    th = threading.Thread(target=run, name="ckpt-guard-hub-verify", daemon=True); th.start(); th.join(timeout)
    if th.is_alive(): return None, f"hub_timeout_{timeout}s"
    if "err" in box: return None, "hub_error: " + box["err"]
    hits = [i for i in box.get("infos", []) if getattr(i, "path", None) == path_in_repo]
    if not hits: return None, "hub_remote_missing"
    lfs = getattr(hits[0], "lfs", None)
    sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs is not None else None
    return {"size": getattr(hits[0], "size", None), "sha256": sha}, None


_PRUNE_HUB_INFO_FN = _hub_remote_info  # indirection for offline tests
_PRUNE_VERIFY_EVERY_S = float(os.environ.get("AGILLM_PRUNE_VERIFY_EVERY_S", "300"))
_PRUNE_VERIFY_MAX_AGE_S = float(os.environ.get("AGILLM_PRUNE_VERIFY_MAX_AGE_S", "7200"))
_PRUNE_VERIFIED = {}          # str(path) -> entry; replaced atomically by the verifier thread
_PRUNE_VERIFIER = {"thread": None, "last_refresh_ts": None, "last_error": None}


# v43c NEVER-DELETE: Standard 2975754 refuges (until its HF upload finishes) + any hardlink sharing an inode with them.
_CK_NEVER_DELETE = ("/workspace/dual_training_20260920/ckpt_hardlink_refuge_20260925/", "/workspace/dual_training_20260920/hf_upload_pin_2975754/",
                    "/workspace/gb10_nvfp4_openai_20260915/live_canary/", "/workspace/hf_hold_upload_pin/", "/workspace/dual_training_20260920/")
def _ck_never_delete(p):
    try:
        rp = os.path.realpath(str(p)) + ("/" if os.path.isdir(str(p)) else "")
        if any(rp.startswith(x) or (rp + "/").startswith(x) for x in _CK_NEVER_DELETE): return True
        st = os.stat(str(p))
        if st.st_nlink > 1: return True
        return False
    except OSError:
        return True


def _prune_candidates(sd):
    out = []
    for p in Path(sd).iterdir():
        if p.suffix not in (".pt", ".ckpt") or "DO_NOT_PRUNE" in p.name or "NONCANONICAL" in p.name: continue
        mt = _ck_re.search(r"step0*(\d+)", p.name)
        if mt: out.append((int(mt.group(1)), p))
    return sorted(out)


def _prune_verify_refresh(sd, protect=()):
    """One verifier pass (background thread only; may block on the network). Builds the
    verified-deletable list: receipted (ok:true) freezes whose Hub copy exists at the receipt's
    remote path with matching size (and LFS sha256 when both sides have it). Skips st_nlink>1."""
    sd = Path(sd); new = {}
    prot = set()
    for p in protect:
        try: prot.add(Path(p).resolve().stat().st_ino)
        except OSError: pass
    for cstep, p in _prune_candidates(sd):
        try: st = p.stat()
        except OSError: continue
        if p.is_symlink() or not p.is_file() or st.st_nlink > 1 or st.st_ino in prot: continue
        rec, rpath = _load_receipt(sd, cstep)
        if rec is None: continue
        loc = rec.get("local") or rec.get("source")
        if loc and Path(loc).name != p.name: continue
        if rec.get("size") is not None and int(rec["size"]) != st.st_size: continue
        repo = rec.get("repo") or _PRUNE_REPO
        hpath = rec.get("path_in_repo") or rec.get("remote") or rec.get("hub_path")
        if not hpath or repo != _PRUNE_REPO: continue
        info, err = _PRUNE_HUB_INFO_FN(repo, hpath, _PRUNE_HUB_TIMEOUT_S)
        ok = info is not None and info.get("size") == st.st_size and not (info.get("sha256") and rec.get("sha256") and info["sha256"] != rec["sha256"])
        if not ok:
            try: _emit({"event": "checkpoint_guard_prune_refused", "path": str(p), "step": cstep, "receipt": str(rpath), "hub_path": hpath,
                        "local_size": st.st_size, "hub": info, "error": err or "hub_size_or_sha_mismatch"})
            except Exception: pass
            continue
        new[str(p)] = {"path": str(p), "step": cstep, "size": st.st_size, "mtime": st.st_mtime, "ino": st.st_ino, "dev": st.st_dev,
                       "verified_at": time.time(), "receipt": str(rpath), "hub_repo": repo, "hub_path": hpath,
                       "hub_size": info.get("size"), "hub_sha256": info.get("sha256"), "sha_compared": bool(info.get("sha256") and rec.get("sha256"))}
    global _PRUNE_VERIFIED
    _PRUNE_VERIFIED = new
    _PRUNE_VERIFIER["last_refresh_ts"] = time.time()
    return new


def _start_prune_verifier(sd, protect=()):
    """Daemon thread refreshing the verified-deletable list every _PRUNE_VERIFY_EVERY_S. Never touches the GPU."""
    if _PRUNE_VERIFIER["thread"] is not None: return _PRUNE_VERIFIER["thread"]
    def loop():
        while True:
            try: _prune_verify_refresh(sd, protect); _PRUNE_VERIFIER["last_error"] = None
            except Exception as e: _PRUNE_VERIFIER["last_error"] = f"{type(e).__name__}: {e}"[:300]
            time.sleep(_PRUNE_VERIFY_EVERY_S)
    th = threading.Thread(target=loop, name="ckpt-guard-prune-verifier", daemon=True); th.start()
    _PRUNE_VERIFIER["thread"] = th
    return th


def _path_in_use(path):
    """True if any process has `path` open, or an upload process names it on its command line."""
    path = Path(path); sp = str(path); name = path.name
    try: real = os.path.realpath(sp)
    except OSError: real = sp
    me = os.getpid()
    for pd in Path("/proc").iterdir():
        if not pd.name.isdigit() or int(pd.name) == me: continue
        try:
            cmd = (pd / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            if (sp in cmd or name in cmd) and ("upload" in cmd or "huggingface" in cmd or "hf_transfer" in cmd): return True
        except OSError: pass
        try:
            for fd in (pd / "fd").iterdir():
                try:
                    if os.readlink(fd) in (sp, real): return True
                except OSError: pass
        except OSError: pass
    return False


def _prune_log(entry):
    entry = {"ts": time.time(), "pid": os.getpid(), "lane": _PRUNE_LANE, **entry}
    try:
        with open(_PRUNE_LOG_PATH, "a") as fh:
            fh.write(json.dumps(entry) + "\n"); fh.flush()
            try: os.fsync(fh.fileno())
            except OSError: pass
    except Exception: pass
    try: _emit({"event": "checkpoint_guard_prune_delete", **entry})
    except Exception: pass


def _load_receipt(sd, step):
    p = Path(sd) / f"lfs_upload_receipt_{step}.json"
    try: r = json.loads(p.read_text())
    except Exception: return None, p
    return (r if isinstance(r, dict) and r.get("ok") is True else None), p


def _ckpt_free_space(sd, need, protect, ckpt_state=None, step=None, reason="disk_precheck"):
    """Free space in `sd` until free >= need. Returns (ok, detail). Never raises."""
    sd = Path(sd); t_end = time.monotonic() + _PRUNE_DEADLINE_S
    def free(): return shutil.disk_usage(sd).free
    try:
        if free() >= need: return True, "already_enough"
        prot_ino = set(); prot_path = set()
        for p in protect:
            p = Path(p)
            for q in (p, p.resolve() if p.exists() else p):
                prot_path.add(str(q))
                try: prot_ino.add((q.stat().st_dev, q.stat().st_ino))
                except OSError: pass
        try: lk = open(_PRUNE_LOCK_PATH, "a+")
        except OSError as e: return False, f"prune_lock_open_failed: {e}"
        try:
            t_lock = time.monotonic() + _PRUNE_LOCK_TIMEOUT_S; got = False
            while True:
                try: _ck_fcntl.flock(lk.fileno(), _ck_fcntl.LOCK_EX | _ck_fcntl.LOCK_NB); got = True; break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= t_lock: break
                    time.sleep(0.5)
            if not got: return False, "prune_lock_busy"
            try:
                deleted = 0
                def safe_kind(p):
                    try: st = p.stat()
                    except OSError: return None
                    if str(p) in prot_path or str(p.resolve()) in prot_path or (st.st_dev, st.st_ino) in prot_ino: return None
                    if p.is_symlink() or not p.is_file(): return None
                    return st
                # (a) this trainer's own leftover .tmp partials
                roots = tuple(Path(x).name for x in protect)
                for p in sorted(sd.iterdir()):
                    if free() >= need: return True, f"freed_deleted_{deleted}"
                    if ".tmp" not in p.name or not p.name.startswith(roots): continue
                    st = safe_kind(p)
                    if st is None or _path_in_use(p): continue
                    if st.st_nlink > 1: continue
                    if _ck_never_delete(p): continue
                    f0 = free()
                    try: p.unlink()
                    except OSError: continue
                    deleted += 1
                    _prune_log({"free_gain_bytes": free() - f0, "path": str(p), "size": st.st_size, "step": step, "local_receipt": None, "hub_repo": None,
                                "hub_path": None, "hub_size": None, "hub_sha256": None, "kind": "tmp_partial", "reason": f"{reason}: leftover partial"})
                # (b) receipted freezes the BACKGROUND verifier confirmed on the Hub within
                # _PRUNE_VERIFY_MAX_AGE_S, unchanged since (size, mtime, inode), st_nlink == 1. No network here.
                verified = _PRUNE_VERIFIED; now = time.time()
                for cstep, p in _prune_candidates(sd):
                    if free() >= need: return True, f"freed_deleted_{deleted}"
                    if time.monotonic() >= t_end: return free() >= need, "prune_deadline"
                    st = safe_kind(p)
                    if st is None or st.st_nlink > 1: continue  # hardlinked (latest.pt / hf_hold_upload_pin): frees nothing
                    ent = verified.get(str(p))
                    if ent is None or now - ent["verified_at"] > _PRUNE_VERIFY_MAX_AGE_S: continue
                    if (ent["size"], ent["mtime"], ent["ino"]) != (st.st_size, st.st_mtime, st.st_ino): continue
                    rec, rpath = _load_receipt(sd, cstep)
                    if rec is None: continue  # receipt must still exist and be ok:true
                    if _path_in_use(p): continue
                    if _ck_never_delete(p): continue
                    f0 = free()
                    try: p.unlink()
                    except OSError: continue
                    gain = free() - f0
                    deleted += 1
                    if ckpt_state is not None: ckpt_state["ckpt_prune_deletions"] = ckpt_state.get("ckpt_prune_deletions", 0) + 1
                    _prune_log({"path": str(p), "size": st.st_size, "step": cstep, "local_receipt": str(rpath), "hub_repo": ent["hub_repo"],
                                "hub_path": ent["hub_path"], "hub_size": ent["hub_size"], "hub_sha256": ent["hub_sha256"],
                                "sha_compared": ent["sha_compared"], "verified_at": ent["verified_at"], "free_gain_bytes": gain,
                                "kind": "receipted_checkpoint", "reason": reason})
                ok = free() >= need
                return ok, (f"freed_deleted_{deleted}" if ok else f"insufficient_disk_nothing_safe_to_delete(deleted={deleted})")
            finally:
                try: _ck_fcntl.flock(lk.fileno(), _ck_fcntl.LOCK_UN)
                except OSError: pass
        finally:
            lk.close()
    except Exception as e:
        return False, f"prune_error: {type(e).__name__}: {e}"[:300]


def _guarded_checkpoint(inner, ckpt_state, tmp, writer, step, reason, *a, free_fn=None, **kw):
    """Run one checkpoint emission; never raises for save errors. True=saved/started, False=failed/skipped.
    On disk-full it calls free_fn() once (delete-then-retry) and retries the save once."""
    if ckpt_state is not None: ckpt_state.pop("_cur_skip_reason", None)
    for attempt in (0, 1):
        try:
            ok = inner(*a, **kw)
        except _CKPT_CATCH as e:
            if _is_cuda_fault(e): raise  # CUDA faults must still stop the run
            tb = traceback.format_exc()
            th = getattr(writer, "_thread", None) if writer is not None else None
            if th is None or not th.is_alive():
                try: Path(tmp).unlink()
                except OSError: pass
            if ckpt_state is not None:
                ckpt_state["saved_clock"] = -1; ckpt_state["retry"] = False  # signal-exit still retries; no per-step retry storm
            err = f"{type(e).__name__}: {e}"
            if attempt == 0 and free_fn is not None and _is_disk_full(e, (ckpt_state or {}).get("last_need_bytes"), Path(tmp).parent):
                freed, detail = free_fn()
                try: _emit({"event": "checkpoint_guard_enospc", "ts": time.time(), "step": step, "error": err[:300], "freed": freed, "detail": detail})
                except Exception: pass
                if freed: continue  # lock already released inside free_fn; retry the save once
                if ckpt_state is not None: ckpt_state["ckpt_last_skip_reason"] = detail
                _ckpt_guard_note_failure(ckpt_state, step, reason, "enospc_skip", f"{err} | {detail}", tb)
                return False
            _ckpt_guard_note_failure(ckpt_state, step, reason, "exception" if attempt == 0 else "exception_after_prune_retry", err, tb)
            return False
        if ok is False:
            detail = (ckpt_state.pop("_cur_skip_reason", None) if ckpt_state is not None else None) or "save skipped"
            if ckpt_state is not None: ckpt_state["ckpt_last_skip_reason"] = detail
            _ckpt_guard_note_failure(ckpt_state, step, reason, "skipped", detail)
        return ok
    return False


def _rpv16_pick_newest_resume(requested, save_dir):
    """v43: resume from the NEWEST valid checkpoint (highest step) among the requested path and the
    save dir's resumable/freeze/follow files. Size-checked, step read from a matching sidecar or an
    mmap reload, and the winner is reloaded to confirm its step. Never picks an older copy."""
    try:
        sd = Path(save_dir); cands = set()
        if requested: cands.add(Path(requested))
        for pat in ("*current-resumable.pt", "freeze_step*_v6.pt", "*follow*.pt", "latest.pt"):
            cands.update(sd.glob(pat))
        min_size = int(os.environ.get("AGILLM_RESUME_MIN_BYTES", str(10 * 1024**3)))
        seen_real = {}; rejected = []
        for p in cands:
            try: rp = p.resolve(); st = rp.stat()
            except OSError: continue
            if str(rp) in seen_real or ".tmp" in rp.name: continue
            if st.st_size < min_size: rejected.append((str(rp), "too_small")); continue
            step = None
            try:
                m = json.loads(rp.with_suffix(rp.suffix + ".stepmeta.json").read_text())
                if m.get("ino") == st.st_ino and m.get("size") == st.st_size: step = int(m["step"])
            except Exception: pass
            if step is None:
                try: step = int(torch.load(rp, map_location="cpu", mmap=True, weights_only=False)["step"])
                except Exception as e: rejected.append((str(rp), f"load:{type(e).__name__}")); continue
            seen_real[str(rp)] = (step, st.st_mtime)
        order = sorted(seen_real.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)
        for path, (step, _mt) in order:
            try: real = int(torch.load(path, map_location="cpu", mmap=True, weights_only=False)["step"])
            except Exception as e: rejected.append((path, f"confirm_load:{type(e).__name__}")); continue
            if real != step: rejected.append((path, f"confirm_step_mismatch:{real}!={step}")); continue
            _emit({"event": "resume_pick_newest", "requested": str(requested), "chosen": path, "step": real,
                   "candidates": {k: v[0] for k, v in seen_real.items()}, "rejected": rejected})
            return path
        _emit({"event": "resume_pick_newest", "requested": str(requested), "chosen": str(requested), "step": None, "candidates": {}, "rejected": rejected, "note": "no valid candidate; falling back to requested"})
    except Exception as e:
        try: _emit({"event": "resume_pick_newest_failed", "error": f"{type(e).__name__}: {e}"[:300]})
        except Exception: pass
    return requested


def _write_checkpoint(payload, tmp, dst, fsync):
    try:
        _emit({"event":"checkpoint_write_enter","tmp":str(tmp),"dst":str(dst),"fsync":bool(fsync),"step":(payload.get("step") if isinstance(payload,dict) else None),"tmp_abs":str(Path(tmp).resolve()),"dst_abs":str(Path(dst).resolve())})
    except Exception as _e:
        pass
    # v1 sequence (torch.save to latest.pt.tmp, atomic os.replace). With fsync the data is
    # durable before the rename and the directory entry after it.
    # v43h: verify write actually lands (live SIGTERM saves were emitting checkpoint events
    # without updating dst mtime/step — catch that closed).
    tmp = Path(tmp); dst = Path(dst)
    pre_ino = dst.stat().st_ino if dst.exists() else None
    pre_mtime = dst.stat().st_mtime if dst.exists() else None
    try:
        torch.save(payload, tmp)
        if not tmp.exists() or tmp.stat().st_size < _CKPT_GUARD_MIN_BYTES:
            raise RuntimeError(f"checkpoint tmp missing/too_small after torch.save: {tmp} exists={tmp.exists()} size={tmp.stat().st_size if tmp.exists() else None}")
        if fsync and os.name == "posix":
            fd = os.open(str(tmp), os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        if _CKPT_VERIFY_RELOAD and isinstance(payload, dict) and "step" in payload:
            # v84: reload-verify (mmap: tensors stay on disk) BEFORE the new file replaces latest.
            _v = torch.load(tmp, map_location="cpu", mmap=True, weights_only=False)
            _vs = _v.get("step") if isinstance(_v, dict) else None
            del _v
            if _vs != payload.get("step"):
                raise RuntimeError(f"checkpoint reload-verify failed: tmp step={_vs} expected={payload.get('step')}")
        os.replace(tmp, dst)
    except BaseException:
        # v84 save guard: never leave a partial latest.pt.tmp behind (keeper refuses to launch on it).
        try: tmp.unlink()
        except OSError: pass
        raise
    if fsync: _fsync_best_effort(Path(dst).parent)
    st = dst.stat()
    if pre_mtime is not None and st.st_mtime == pre_mtime and st.st_ino == pre_ino:
        raise RuntimeError(f"checkpoint replace did not change dst inode/mtime: {dst} ino={st.st_ino} mtime={st.st_mtime}")
    # sidecar meta for cheap verification without loading 11GB
    try:
        meta = {"step": payload.get("step"), "seen_tokens": payload.get("seen_tokens"), "optimizer_update_clock": payload.get("optimizer_update_clock"), "dst": str(dst), "size": st.st_size, "mtime": st.st_mtime, "ino": st.st_ino}
        side = dst.with_suffix(dst.suffix + ".stepmeta.json")
        side.write_text(json.dumps(meta) + "\n")
    except Exception as _e:
        # Sidecar metadata is advisory. The checkpoint bytes have already been atomically
        # installed, so a sidecar failure must never turn a successful save into a trainer crash.
        try:
            _emit({"event": "checkpoint_sidecar_warning", "ts": time.time(),
                   "path": str(dst), "error": f"{type(_e).__name__}: {_e}"[:300]})
        except Exception:
            pass


def _write_checkpoint_nonfatal(payload, tmp, dst, fsync, *, step=None, reason="periodic"):
    """Attempt checkpoint emission without killing training on storage/I/O failure.

    Atomic tmp->dst replacement means the previous complete checkpoint remains authoritative
    unless the new write fully succeeds. Any partial tmp is removed. We deliberately catch
    Exception, not BaseException, so KeyboardInterrupt/SystemExit still retain normal semantics.
    """
    try:
        _write_checkpoint(payload, tmp, dst, fsync)
        return True
    except Exception as e:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            free_now = shutil.disk_usage(Path(dst).parent).free
        except Exception:
            free_now = None
        try:
            _emit({"event": "checkpoint_write_failed_nonfatal", "ts": time.time(),
                   "path": str(dst), "tmp": str(tmp), "step": step, "reason": reason,
                   "error": f"{type(e).__name__}: {e}"[:400],
                   "free_bytes": free_now,
                   "action": "training_continues; previous complete checkpoint preserved; retry on a later save boundary"})
        except Exception:
            pass
        return False


class _CkptWriter:
    """Single-slot background checkpoint writer: never more than one write in flight.

    Non-daemon on purpose: on any orderly interpreter exit the in-flight write completes, so
    latest.pt.tmp is not left behind (the keeper refuses to launch while it exists). A failed
    write removes its own tmp. Only a hard kill (SIGKILL/OOM/power) mid-write can strand
    latest.pt.tmp, exactly as it can during v1's synchronous save.
    """
    def __init__(self):
        self._thread = None
        self._result = None

    def start(self, payload, tmp, dst, event, fsync, extra, t_req):
        if self._thread is not None: raise RuntimeError("checkpoint write already in flight")
        box = [payload]
        self._thread = threading.Thread(target=self._run, args=(box, tmp, dst, dict(event), fsync, extra, t_req), name="ckpt-writer", daemon=False)
        self._thread.start()

    def _run(self, box, tmp, dst, event, fsync, extra, t_req):
        t0 = time.perf_counter()
        try:
            _write_checkpoint(box[0], tmp, dst, fsync)
            if extra is not None:
                now = time.perf_counter()
                event.update({"ts": time.time(), **extra, "write_s": now - t0, "save_s": now - t_req})
            self._result = {"ok": True, "event": event}
        except BaseException as e:
            try: os.unlink(tmp)
            except OSError: pass
            self._result = {"ok": False, "event": event, "error": f"{type(e).__name__}: {e}"[:240], "traceback": traceback.format_exc()[-4000:]}
        finally:
            box.clear()  # release the host snapshot as soon as the file is in place

    def finish(self, block):
        th = self._thread
        if th is None or (not block and th.is_alive()): return None
        th.join(); self._thread = None
        r, self._result = self._result, None
        return r


def _report_ckpt(result, ckpt_state, model=None, seen_tokens=None):
    # Main thread only. The "checkpoint" event is published after os.replace, so a log
    # follower never sees it before latest.pt is the new file.
    if result is None: return
    if result["ok"]:
        _emit(result["event"])
        _ckpt_guard_note_success(ckpt_state, (result.get("event") or {}).get("step"))
        # Triple-goal v6: restore inproc heldout CE so promote/prepare can advance without a third CUDA process.
        try:
            if model is not None:
                ev = result.get("event") or {}
                step = ev.get("step")
                st = seen_tokens if seen_tokens is not None else ev.get("seen_tokens")
                if step is not None and _rpv16_inproc_heldout is not None and hasattr(_rpv16_inproc_heldout, "after_save"):
                    _rpv16_inproc_heldout.after_save(model, step, st if st is not None else 0, emit=_emit, ckpt_path=ev.get("path"))
        except Exception as _e:
            try:
                _emit({"event": "heldout_inproc_failed", "error": f"{type(_e).__name__}: {_e}"[:300]})
            except Exception:
                pass
        return
    if ckpt_state is not None:
        ckpt_state["async_disabled"] = True; ckpt_state["retry"] = True; ckpt_state["saved_clock"] = -1
    ev = result["event"]
    _ckpt_guard_note_failure(ckpt_state, ev.get("step"), "async", "async_write", result.get("error"), result.get("traceback"))
    _emit({"event": "checkpoint_async_failed", "ts": time.time(), "path": ev.get("path"), "step": ev.get("step"), "error": result["error"], "action": "tmp removed; async disabled for this process; synchronous retry at the next update boundary"})


def _shutdown_background():
    # Any exit path from train(): stop the producer and let an in-flight write finish.
    stream = _BACKGROUND["stream"]; writer = _BACKGROUND["writer"]; flush = _BACKGROUND["flush"]
    if flush is not None:
        try: flush(True)  # best effort: deferred step records of a run that is dying for another reason
        except Exception: pass
    if stream is not None: stream.stop_prefetch()
    if writer is not None: _report_ckpt(writer.finish(block=True), _BACKGROUND["ckpt_state"])


def _rpv16_resume_required(requested):
    """Honor the keeper's fail-closed resume contract before training can start."""
    required = os.environ.get("AGILLM_REQUIRE_RESUME", "0") == "1"
    if required and (not requested or not Path(requested).is_file()):
        _emit({"event": "resume_required_missing", "path": str(requested or ""),
               "reason": "AGILLM_REQUIRE_RESUME=1 requires an existing checkpoint file",
               "action": "refusing fresh-start training"})
        raise SystemExit(3)
    return required


def _checkpoint_periodic_due(ckpt_state, update_clock, every_updates, every_sec, now):
    """Bound periodic attempts by elapsed updates or time since the previous attempt.

    A time-triggered checkpoint also satisfies the update cadence. Keeping the
    attempt clock separate from saved_clock preserves the save guard's bounded
    retries: saved_clock can become -1 after an I/O failure. Signal saves and
    asynchronous failure recovery retain their independent forced-save paths.
    """
    due_updates = bool(every_updates > 0 and
                       update_clock - ckpt_state["last_periodic_attempt_clock"] >= every_updates)
    due_time = every_sec > 0 and now - ckpt_state["last_save_wall"] >= every_sec
    return due_updates, due_time


def train(args):
    if os.environ.get("AGILLM_RESUME_NEWEST", "1") != "0": args.resume = _rpv16_pick_newest_resume(args.resume, args.save_dir)  # v43
    _rpv16_resume_required(args.resume)  # Check before data/model setup.
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sources = list(HF_SOURCES) if args.dataset == "mix" else [args.dataset]
    stream=HFStream(sources,args.seed)
    sources=list(stream.sources)  # Report the actual verified data mix.
    print(json.dumps({"event":"dataset_ready","sources":sources,"tokenizer":str(TOKENIZER_JSON)}),flush=True)
    t0=time.time(); model=build_model(); model.ce_chunk=args.ce_chunk; model.fused_ce=bool(args.fused_ce); model.fused_ce_grad_scale_hint=1.0/float(max(1,args.grad_accum)); torch.cuda.synchronize()
    params=sum(p.numel() for p in model.parameters())
    print(json.dumps({"event":"model_ready","parameters":params,"expected":parameter_accounting()["total"],"build_s":time.time()-t0,"cuda_alloc":torch.cuda.memory_allocated(),"target_alignment":TARGET_ALIGNMENT}),flush=True)
    if params != parameter_accounting()["total"]: raise RuntimeError("parameter count drift")
    def _uniq_params(xs):
        out=[]; seen_ids=set()
        for q in xs:
            if id(q) not in seen_ids:
                seen_ids.add(id(q)); out.append(q)
        return out
    stage_param_sets=[]
    for si in range(STAGES):
        ps=list(model.stages[si].parameters())
        if si == 0:
            ps += list(model.embed.parameters()) + list(model.in_proj.parameters())
        stage_param_sets.append(_uniq_params(ps))
    head_params=_uniq_params(model.head_parameters())
    try:
        import bitsandbytes as bnb
        def _make_opt(ps):
            if RPV16_FP32_MASTER and fm_has_eligible(ps):  # fable v43i: only banks holding registered trunk weights
                return _FP32MasterOptimizer(lambda g: bnb.optim.PagedAdamW8bit(g,lr=args.lr,betas=(0.9,0.95),weight_decay=0.1), ps)
            return bnb.optim.PagedAdamW8bit(ps,lr=args.lr,betas=(0.9,0.95),weight_decay=0.1)
        optimizer_name="paged_adamw8bit_bank"
    except Exception:
        def _make_opt(ps):
            return torch.optim.AdamW(ps,lr=args.lr,betas=(0.9,0.95),weight_decay=0.1,foreach=True)
        optimizer_name="adamw_bank"
    if RPV16_FP32_MASTER:  # fable v43i 2026-10-03: trunk-only fp32 masters; head, embedding and all RMSNorms stay as before
        fm_register_eligible(fm_trunk_params(model)); fm_assert_trunk_only(model)
    stage_opts=[_make_opt(ps) for ps in stage_param_sets]
    if RPV16_TIED_HEAD:
        # 2026-09-23 cowork: no weight decay on the classifier bias (log-unigram prior).
        _tied_nd = [model.tied_bias]
        head_opt=_make_opt([{"params":[p for p in head_params if all(p is not q for q in _tied_nd)]},{"params":_tied_nd,"weight_decay":0.0}])
    else:
        head_opt=_make_opt(head_params)
    _tied_migration=False
    print(json.dumps({"event":"fp32_master_weights","schema":"fable.rpv16.fp32-master.v2",**fm_telemetry()}),flush=True)
    AH=allheads_setup(args,model,head_params)
    seen=0; step_times=[]; start_step=1; optimizer_update_clock=0; micro_in_update=0
    # Recheck after setup; a disappearing required file must never fall through to fresh training.
    if _rpv16_resume_required(args.resume) or (args.resume and Path(args.resume).exists()):
        t_resume0=time.perf_counter(); stream_restore_s=None
        _fm_resume_file_stat = os.stat(args.resume)
        ck=torch.load(args.resume,map_location="cpu",weights_only=False)
        ckpt_load_s=time.perf_counter()-t_resume0
        missing, unexpected = model.load_state_dict(ck["model"],strict=False)
        _RPV_SPARSE_TOPOLOGY.restore_sparse_topology(model, SparseLinear, ck, emit=_emit)
        if missing or unexpected:
            print(json.dumps({"event":"model_resume_non_strict","missing":missing[:16],"unexpected":unexpected[:16]}),flush=True)
        if RPV16_TIED_HEAD and "tied_bias" in missing:
            # 2026-09-23 cowork: RPV16-head checkpoint -> tied classifier. Bias = log unigram, scale calibrated on the
            # first batch, then TIED_WARMUP_UPDATES head-only updates before joint training resumes.
            _tied_migration=True
            with torch.no_grad():
                _lp=torch.load(TIED_UNIGRAM_PATH,map_location="cpu",weights_only=False)
                _lp=_lp["logp"] if isinstance(_lp,dict) else _lp
                if tuple(_lp.shape)!=(VOCAB,) or not bool(torch.isfinite(_lp).all()):
                    raise RuntimeError("tied classifier unigram prior has the wrong shape or nonfinite values")
                model.tied_bias.copy_(_lp.to(device=model.tied_bias.device,dtype=torch.float32))
                model.tied_logit_scale.fill_(1.0)
                model.tied_warmup_done.zero_()
            model._tied_needs_calibration=True
            print(json.dumps({"event":"tied_head_migration","from":"rpv16_rank16_product_vocab","bias_init":TIED_UNIGRAM_PATH,"warmup_updates":TIED_WARMUP_UPDATES,"warmup_lr":TIED_WARMUP_LR}),flush=True)
        if "factor_head.weight" in missing or "mix_head.weight" in missing:
            with torch.no_grad():
                model.factor_head.weight.mul_(0.10)
                model.mix_head.weight.zero_()
            print(json.dumps({"event":"rpv16_affine_near_uniform_init","factor_scale":0.10,"mix":"zero"}),flush=True)
        resume_target_alignment=ck.get("target_alignment")
        bank=ck.get("optimizer_bank")
        _fm_prepare_checkpoint_resume(ck, stage_opts, args.resume, _fm_resume_file_stat)
        if resume_target_alignment != TARGET_ALIGNMENT:
            bank=None
            print(json.dumps({"event":"optimizer_target_migration_reset","from":resume_target_alignment or "legacy_double_shift_i_plus_2","to":TARGET_ALIGNMENT,"reason":"objective alignment changed; stale Adam moments rejected"}),flush=True)
        if isinstance(bank,dict) and isinstance(bank.get("stages"),list) and len(bank["stages"]) == STAGES:
            try:
                for o,sd in zip(stage_opts,bank["stages"]): o.load_state_dict(sd)
                if _tied_migration:
                    head_state = "fresh_tied_classifier"      # 2026-09-23 cowork: saved state belonged to the RPV16 head
                else:
                    try:
                        head_opt.load_state_dict(bank["head"])
                        head_state = "resumed"
                    except Exception as e:
                        raise RuntimeError("head optimizer resume failed; refusing reset on exact production lineage") from e
                print(json.dumps({"event":"optimizer_bank_resumed","optimizer":optimizer_name,"head_state":head_state}),flush=True)
            except Exception as e:
                raise RuntimeError("optimizer bank resume failed; refusing partial-state training: " + str(e)[:240]) from e
        elif "optimizer" in ck:
            print(json.dumps({"event":"optimizer_migration_reset","from":ck.get("optimizer_name","legacy"),"to":optimizer_name,"reason":"static optimizer bank migration"}),flush=True)
        if AH is not None: AH.load_from_ckpt(ck)
        seen=int(ck.get("seen_tokens",0)); ck_step=int(ck.get("step",0)); start_step=ck_step+1
        if "optimizer_update_clock" in ck:
            optimizer_update_clock=int(ck["optimizer_update_clock"])
            micro_in_update=int(ck.get("micro_in_update",0))
            if micro_in_update != 0:
                raise RuntimeError("checkpoint must be saved at an optimizer-update boundary")
            phase_clock_source="checkpoint"
            resume_grad_accum=int(ck.get("grad_accum",args.grad_accum))
        else:
            legacy_ga=int(args.legacy_resume_grad_accum)
            if legacy_ga < 1 or ck_step % legacy_ga:
                raise RuntimeError(f"cannot derive legacy optimizer clock: step={ck_step} legacy_grad_accum={legacy_ga}")
            optimizer_update_clock=ck_step // legacy_ga
            micro_in_update=0
            phase_clock_source="legacy_derived"
            resume_grad_accum=legacy_ga
        if isinstance(ck.get("data_state"),dict):
            t_stream0=time.perf_counter()
            stream.load_state_dict(ck["data_state"])
            stream_restore_s=time.perf_counter()-t_stream0
            ds=ck["data_state"]
            print(json.dumps({"event":"data_cursor_resumed","schema":ds.get("schema"),"rows_emitted":ds.get("rows_emitted",{}),"token_buffer":int(ds.get("token_buffer",torch.empty(0)).numel())}),flush=True)
        else:
            print(json.dumps({"event":"data_cursor_migration_reset","reason":"legacy checkpoint lacks exact stream cursor","checkpoint_step":ck_step,"seen_tokens":seen,"note":"one-time data-order reset; future v4 checkpoints resume exactly"}),flush=True)
        print(json.dumps({"event":"resumed","path":args.resume,"step":start_step-1,"seen_tokens":seen,"optimizer_update_clock":optimizer_update_clock,"phase":optimizer_update_clock%(STAGES+1),"phase_clock_source":phase_clock_source,"resume_grad_accum":resume_grad_accum,"requested_grad_accum":args.grad_accum,"target_alignment":TARGET_ALIGNMENT,"resume_target_alignment":resume_target_alignment,**({} if args.v1_telemetry else {"ts":time.time(),"ckpt_load_s":ckpt_load_s,"stream_restore_s":stream_restore_s,"resume_total_s":time.perf_counter()-t_resume0,"proc_uptime_s":time.time()-_PROC_T0})}),flush=True)
        del ck
        import gc; gc.collect(); torch.cuda.empty_cache()
    train_stage = None
    global_anchor = False
    active_opt = None
    active_params = None
    # ---- v2 full-stack host path. Nothing here alters batches, RNG use, loss, gradients,
    # optimizer updates, the phase clock, or checkpoint contents/schema.
    lazy = not args.strict_step_sync
    tel2 = not args.v1_telemetry
    ckpt_writer = _CkptWriter()
    ckpt_state = {"async_disabled": False, "retry": False, "saved_clock": optimizer_update_clock, "last_save_wall": time.time(),
                  "last_periodic_attempt_clock": optimizer_update_clock,
                  "ckpt_save_failures_consecutive": 0, "ckpt_save_failures_total": 0, "ckpt_last_error": None, "ckpt_last_error_ts": None,
                  "status_path": str(Path(args.save_dir) / "ckpt_save_status.json")}
    _ckpt_every_sec = float(os.environ.get("AGILLM_CKPT_EVERY_SEC", "1200"))
    _BACKGROUND.update({"stream": stream, "writer": ckpt_writer, "ckpt_state": ckpt_state})
    _rolling_ckpt = Path(args.save_dir) / "RPV16-GB10-1PF-v6-current-resumable.pt"
    _compat_latest = Path(args.save_dir) / "latest.pt"
    if os.environ.get("AGILLM_PRUNE_VERIFIER", "1") != "0":
        try: _start_prune_verifier(args.save_dir, protect=[_rolling_ckpt, _compat_latest])  # daemon; network only off the save path
        except Exception as _e: _emit({"event": "prune_verifier_start_failed", "error": f"{type(_e).__name__}: {_e}"[:300]})

    def _uk_stamp():
        return datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%dT%H%M%S%z")

    pending = deque()
    # Full pipeline rates count only train records whose CUDA completion event has
    # completed. Host enqueue timestamps and GPU event durations are not wall time.
    _pipeline_start = time.perf_counter()
    _pipeline_seen0 = seen
    _pipeline_targets = 0

    def _finalize_pipeline_telemetry(rec):
        nonlocal _pipeline_targets
        _elapsed = time.perf_counter() - _pipeline_start
        _pipeline_targets += int(rec.get("n_targets", args.batch * args.seq))
        rec.update({
            "data_pipeline": "astra-speedstack-completed-cursor-v1",
            "pipeline_tok_s": (int(rec["seen_tokens"]) - _pipeline_seen0) / _elapsed,
            "pipeline_target_s": _pipeline_targets / _elapsed,
            "wall_time_s": _elapsed,
            "data_prefetch_qsize": rec.get("prefetch_q", stream.last_qsize),
        })

    def _flush_train_log(block):
        # Deferred per-step record + NaN/Inf guard. Same scalars v1 checked, read from async
        # host copies once the step's end event has completed. Blocks only when asked to
        # (defer bound reached, or before any state capture).
        while pending:
            rec,host,ev0,ev1=pending[0]
            if block: ev1.synchronize()
            elif not ev1.query(): return
            pending.popleft()
            loss_v,aux_v,gn_v=host.tolist()
            bad="nonfinite loss" if not math.isfinite(loss_v) else ("nonfinite grad norm" if rec["optimizer_step"] and not math.isfinite(gn_v) else None)
            if bad:
                pending.clear()  # as in v1, nothing at or after the bad step is ever logged or saved
                raise RuntimeError(f"{bad} (step {rec['step']})")
            dt=ev0.elapsed_time(ev1)/1000.0
            step_times.append(dt)
            rec["loss"]=loss_v; rec["aux"]=aux_v; rec["grad_norm"]=gn_v if rec["optimizer_step"] else None
            rec["step_s"]=dt; rec["tok_s"]=args.batch*args.seq/dt
            _finalize_pipeline_telemetry(rec)
            _emit(rec)

    _BACKGROUND["flush"]=_flush_train_log

    def _ckpt_free_fn(step):
        dst=_rolling_ckpt
        need=ckpt_state.get("last_need_bytes") or ((dst.stat().st_size if dst.exists() else 12*1024**3)+1024**3)
        return _ckpt_free_space(Path(args.save_dir), need, [dst,_compat_latest], ckpt_state, step, reason="enospc")

    def _checkpoint(step,seen_tokens,force_sync=False,reason="periodic"):
        # v84: every checkpoint emission goes through the guard; a save failure never leaves train().
        return _guarded_checkpoint(lambda _s, _t, force_sync=False: _checkpoint_unguarded(_s, _t, force_sync=force_sync, reason=reason), ckpt_state, Path(args.save_dir)/(_rolling_ckpt.name+".tmp"), ckpt_writer, step, reason, step, seen_tokens, force_sync=force_sync, free_fn=lambda: _ckpt_free_fn(step))  # v43c: reason bound in closure (v43b passed it twice -> TypeError)

    def _checkpoint_unguarded(step,seen_tokens,force_sync=False,reason="periodic"):
        # The save block, shared by periodic update-count and signal/retry paths.
        # it. Callers guarantee an optimizer-update boundary and a passed NaN/Inf guard.
        t_req=time.perf_counter()
        _report_ckpt(ckpt_writer.finish(block=True),ckpt_state, model=model)  # never more than one write in flight
        sd=Path(args.save_dir); sd.mkdir(parents=True,exist_ok=True)
        dst=_rolling_ckpt; tmp=sd/(dst.name+".tmp")
        # Auto-prune only an incomplete stale temp file from this exact save dir.
        # Complete checkpoints are never deleted here; post-upload pruning is handled separately.
        if tmp.exists():
            try:
                age=time.time()-tmp.stat().st_mtime
                if age >= 300:
                    tmp.unlink()
                    _emit({"event":"checkpoint_autoprune","path":str(tmp),"reason":"stale_tmp","age_s":age})
            except FileNotFoundError:
                pass
        expected = dst.stat().st_size if dst.exists() else 12*1024**3
        free = shutil.disk_usage(sd).free
        need = max(expected + 1024**3, int(float(os.environ.get("AGILLM_CKPT_FREE_FLOOR_GB", "25")) * 1024**3))  # v43: 25G free floor
        ckpt_state["last_need_bytes"] = need
        if free < need:
            # v84 delete-then-retry: free space in this same step (shared lock, receipted+verified only).
            freed, detail = _ckpt_free_space(sd, need, [dst,_compat_latest], ckpt_state, step, reason="disk_precheck")
            free = shutil.disk_usage(sd).free
            if not freed or free < need:
                ckpt_state["_cur_skip_reason"] = detail
                _emit({"event":"checkpoint_skipped_disk","step":step,"free_bytes":free,"need_bytes":need,"path":str(dst),"prune_detail":detail})
                return False
            _emit({"event":"checkpoint_disk_freed","step":step,"free_bytes":free,"need_bytes":need,"prune_detail":detail})
        payload={"schema":("agillm.gb10.1pf.recovery.v5" if AH is None else ALLHEADS_CKPT_SCHEMA),"target_alignment":TARGET_ALIGNMENT,"step":step,"seen_tokens":seen_tokens,"model":model.state_dict(),"optimizer_bank":{"stages":[o.state_dict() for o in stage_opts],"head":head_opt.state_dict()},"optimizer_name":optimizer_name,"seed":args.seed,"optimizer_update_clock":optimizer_update_clock,"grad_accum":args.grad_accum,"micro_in_update":micro_in_update,"data_state":stream.consumed_state_dict(),**({} if AH is None else AH.ckpt_extra())}
        payload["sparse_topology"] = _RPV_SPARSE_TOPOLOGY.export_sparse_topology(model, SparseLinear)
        event={"event":"checkpoint","path":str(dst),"compat_path":str(_compat_latest),"step":step,"seen_tokens":seen_tokens,"schema":("v5" if AH is None else "v6"),"target_alignment":TARGET_ALIGNMENT,"optimizer_update_clock":optimizer_update_clock,"grad_accum":args.grad_accum,"uk_stamp":_uk_stamp(),"schedule_clock":"optimizer_update_clock"}
        use_async=not (args.sync_checkpoint or force_sync or ckpt_state["async_disabled"])
        fallback=None; snap=None; avail=None
        if use_async:
            # Unified memory: the host snapshot competes with the GPU and /dev/shm peers.
            avail=_mem_available_bytes(); need_mem=expected+int(args.async_ckpt_reserve_gb*1024**3)
            if avail is None or avail < need_mem:
                use_async=False; fallback="low_host_memory"
        if use_async:
            try:
                torch.cuda.synchronize()  # one sync: paged optimizer state is host-visible managed memory
                t_snap=time.perf_counter(); snap=_host_snapshot(payload,{}); snapshot_s=time.perf_counter()-t_snap
            except Exception as e:
                use_async=False; snap=None; fallback="snapshot_failed:"+type(e).__name__
        ckpt_state["saved_clock"]=optimizer_update_clock
        if use_async:
            del payload
            stall_s=time.perf_counter()-t_req
            ckpt_writer.start(snap,tmp,dst,event,not args.no_ckpt_fsync,{"async":True,"reason":reason,"stall_s":stall_s,"snapshot_s":snapshot_s} if tel2 else None,t_req)
            if tel2: _emit({"event":"checkpoint_async_started","ts":time.time(),"step":step,"optimizer_update_clock":optimizer_update_clock,"stall_s":stall_s,"snapshot_s":snapshot_s,"mem_available_bytes":avail})
        else:
            # V41: checkpoint emission is best-effort for trainer liveness. A failed periodic
            # or async-retry save must not terminate training; tmp is discarded and the prior
            # complete checkpoint/pointer remains authoritative.
            _write_checkpoint(payload,tmp,dst,False)  # v43: failures propagate to _guarded_checkpoint (logs, tmp cleanup, ENOSPC delete-then-retry, skip+count)
            if force_sync and not args.no_ckpt_fsync: _fsync_best_effort(dst)  # after the rename: keeps the tmp window at v1 length
            if tel2:
                save_s=time.perf_counter()-t_req
                event.update({"ts":time.time(),"async":False,"reason":reason,"stall_s":save_s,"save_s":save_s})
                if fallback: event["async_fallback"]=fallback
            _emit(event)
            _ckpt_guard_note_success(ckpt_state, step)
            # sync-checkpoint path never goes through _report_ckpt for this event; run inproc heldout here.
            try:
                if model is not None and _rpv16_inproc_heldout is not None and hasattr(_rpv16_inproc_heldout, "after_save"):
                    _rpv16_inproc_heldout.after_save(model, step, seen_tokens, emit=_emit, ckpt_path=event.get("path"))
            except Exception as _e:
                try:
                    _emit({"event": "heldout_inproc_failed", "error": f"{type(_e).__name__}: {_e}"[:300]})
                except Exception:
                    pass
        try:
            if _compat_latest.is_symlink():
                _compat_latest.unlink()
            elif _compat_latest.exists():
                # The first migration preserves the previous complete checkpoint separately before restart.
                _compat_latest.unlink()
            _compat_latest.symlink_to(dst.name)
        except Exception as e:
            _emit({"event":"checkpoint_latest_link_warning","error":type(e).__name__,"detail":str(e)[:200],"path":str(_compat_latest),"target":dst.name,"uk_stamp":_uk_stamp(),"ts":time.time()})
        return True

    def _boundary_service(step_done):
        # Update-boundary only. (a) a failed background write -> synchronous retry;
        # (b) lossless SIGTERM/SIGINT: join any in-flight write, save synchronously, stop.
        if lazy: _flush_train_log(True)  # NaN/Inf guard covers every step up to step_done
        t0=time.perf_counter()
        _report_ckpt(ckpt_writer.finish(block=True),ckpt_state, model=model)
        saved=None
        if ckpt_state["saved_clock"]!=optimizer_update_clock:
            saved=_checkpoint(step_done,seen,force_sync=True,reason="signal" if _STOP["n"] else "async_retry")
        ckpt_state["retry"]=False
        if not _STOP["n"]: return False
        _emit({"event":"signal_exit","ts":time.time(),"signal":_STOP["sig"],"signals_received":_STOP["n"],"step":step_done,"seen_tokens":seen,"optimizer_update_clock":optimizer_update_clock,"checkpoint_saved":saved,"already_saved":saved is None,"exit_s":time.perf_counter()-t0})
        stream.stop_prefetch()
        return True

    if args.gil_switch_interval_ms > 0: sys.setswitchinterval(args.gil_switch_interval_ms/1000.0)
    if tel2:
        _emit({"event":"fullstack_config","ts":time.time(),"version":"v2_fullstack_20260919","prefetch":not args.no_prefetch,"prefetch_depth":args.prefetch_depth,"prefetch_pin":not args.no_prefetch_pin,"async_checkpoint":not args.sync_checkpoint,"async_ckpt_reserve_gb":args.async_ckpt_reserve_gb,"ckpt_fsync":not args.no_ckpt_fsync,"signal_save":True,"lazy_step_sync":lazy,"finite_check_every":args.finite_check_every,"proc_uptime_s":time.time()-_PROC_T0})
    # Production invariant: deliberate SIGTERM/SIGINT always checkpoints at the next update boundary.
    signal.signal(signal.SIGTERM,_on_signal); signal.signal(signal.SIGINT,_on_signal)
    if not args.no_prefetch:
        stream.abort_check=lambda: _STOP["n"]>0 and micro_in_update==0
        stream.start_prefetch(args.batch,args.seq,depth=args.prefetch_depth,pin=not args.no_prefetch_pin)
    # Scoped execution-only adapter; checkpoint/model/optimizer schemas remain unchanged.
    import sg_bnb_streamstep_c152b07f as _sg_streamstep
    _sg_installed = sum(_sg_streamstep.install(fm_unwrap(o)) for o in [*stage_opts, head_opt])
    _emit({"event":"optimizer_streamstep_ready","installed_banks":_sg_installed,
           "adapter_sha256":"c152b07f21c87853b235596f25414947ff13735e368c30dc658a71b68a66c86c","default":"stock_until_policy_enabled","ts":time.time()})
    _joint_last = {}
    # 2026-09-23 cowork tied-classifier warm-up state (persisted through the model buffer tied_warmup_done).
    tied_done_py = int(model.tied_warmup_done) if RPV16_TIED_HEAD else 0
    tied_warm_opt = None
    if RPV16_TIED_HEAD and tied_done_py < TIED_WARMUP_UPDATES:
        _tied_nd2 = [model.tied_bias]
        tied_warm_opt = torch.optim.AdamW([{"params":[p for p in head_params if all(p is not q for q in _tied_nd2)]},{"params":_tied_nd2}],
                                          lr=TIED_WARMUP_LR, betas=(0.9, 0.95), weight_decay=0.0)
        print(json.dumps({"event":"tied_head_warmup_pending","done":tied_done_py,"target":TIED_WARMUP_UPDATES}),flush=True)

    def _joint_update(obj, lr_now):
        _sg_streamstep.begin_step()
        # 2026-09-23 cowork joint update: replay the stages (joint_backward) and step each stage optimizer the
        # moment its gradients are complete, with the rotating schedule's per-stage clip thresholds; then the head
        # (AR only; SAT/NAT head steps stay in AH.after_step under --allheads-head-grad). Returns the pre-clip
        # global norm over the 16 stage norms.
        ar = (AH is None or obj == "ar")
        gns = [None] * STAGES
        def _stage_update(j):
            gn = torch.nn.utils.clip_grad_norm_(stage_param_sets[j], 1.0 if ar else AH.nonar_max_norm(j))
            if not lazy and not torch.isfinite(gn): raise RuntimeError(f"nonfinite grad norm (joint stage {j})")
            o = stage_opts[j]
            for pg in o.param_groups: pg["lr"] = lr_now
            o.step()
            o.zero_grad(set_to_none=True)
            model.invalidate_stage(j)
            gns[j] = gn.detach().float()
        joint_backward(model, _stage_update)
        gstack = torch.stack(gns)
        _joint_last.clear()
        if ar:
            hg = None
            if any(p.grad is not None for p in head_params):
                hg = torch.nn.utils.clip_grad_norm_(head_params, 1.0)
                if not lazy and not torch.isfinite(hg): raise RuntimeError("nonfinite head grad norm (joint)")
                for pg in head_opt.param_groups: pg["lr"] = lr_now
                head_opt.step()
            head_opt.zero_grad(set_to_none=True)
            # one host sync per AR update, as the rotating schedule's note_ar_grad_norm already had
            vals = (gstack if hg is None else torch.cat([gstack, hg.detach().float().view(1)])).tolist()
            if AH is not None:
                for j in range(STAGES): AH.note_ar_grad_norm(j, vals[j])
            _joint_last["stage_gn"] = [float(f"{v:.5g}") for v in vals[:STAGES]]
            if hg is not None: _joint_last["head_gn"] = float(f"{vals[STAGES]:.5g}")
        return gstack.norm()

    _fm_first_save_pending = _FM_STATS.get("legacy_migrations", 0) > 0
    t_prev=time.perf_counter()
    for step in range(start_step,args.steps+1):
        if micro_in_update == 0:
            # v40: refresh hot execution-only speed policies for rotating and joint modes.
            _mds_v38.refresh_policy()
            _aq_refresh_policy()
            # Persisted optimizer-update clock decouples stage curriculum from
            # raw microstep numbering, allowing safe grad-accum migrations.
            _phase_mod = STAGES if RPV16_SKIP_HEAD_ONLY else (STAGES + 1)
            phase = optimizer_update_clock % _phase_mod
            if args.force_head_only:
                global_anchor = True
                train_stage = None
            elif tied_warm_opt is not None and tied_done_py < TIED_WARMUP_UPDATES:
                # 2026-09-23 cowork: tied-classifier warm-up -- head-only updates (frozen body and embedding) until the
                # new classifier has caught up with the body's features; joint updates resume afterwards.
                global_anchor = True
                train_stage = None
            elif RPV16_JOINT:
                # 2026-09-23 cowork joint update: all stages + interface + head on every update (no rotation, no
                # head-only anchor). The phase clock still advances and still draws the AR/SAT/NAT objective.
                if args.grad_accum != 1:
                    raise RuntimeError("joint update mode requires --grad-accum 1 (set AGILLM_RPV16_JOINT=0 otherwise)")
                global_anchor = False
                train_stage = JOINT_STAGE
            else:
                # v82f hybrid JOINT_EVERY (milder than EVERY=8 / full JOINT=1)
                _joint_every = int(os.environ.get("AGILLM_RPV16_JOINT_EVERY", "0") or "0")
                if _joint_every > 0 and (optimizer_update_clock % _joint_every == 0):
                    if args.grad_accum != 1:
                        raise RuntimeError("JOINT_EVERY requires --grad-accum 1")
                    global_anchor = False
                    train_stage = JOINT_STAGE
                else:
                    global_anchor = (phase == STAGES)  # receipt-compatible name; this is head-only
                    train_stage = None if global_anchor else (STAGES - 1 - phase)
            if train_stage == JOINT_STAGE:
                _joint_obj = "ar" if AH is None else AH.objective_for(optimizer_update_clock, 0)
                model.set_trainable_joint(head=(_joint_obj == "ar"))
                for _o in stage_opts: _o.zero_grad(set_to_none=True)
                active_opt=head_opt; active_params=head_params
            else:
                model.set_trainable_stage(train_stage, head_anchor=global_anchor)
                if (RPV16_ALWAYS_HEAD and (not global_anchor) and train_stage != JOINT_STAGE):
                    _pre_obj = "ar" if AH is None else AH.objective_for(optimizer_update_clock, micro_in_update)
                    if _pre_obj == "ar":
                        for _p in model.head_parameters():
                            _p.requires_grad_(True)
                        # Output-side tied embedding was only stepped on stage 0 and joint
                        # updates, so the unigram bias moved ~8x more often than the codebook
                        # it has to align with. Give that codebook its CE gradient on the other
                        # AR stage steps and apply it with the existing stage-0 Adam state.
                        if int(train_stage) != 0:
                            model.embed.weight.requires_grad_(True)
                            stage_opts[0].zero_grad(set_to_none=True)
                if global_anchor:
                    _warm = tied_warm_opt is not None and tied_done_py < TIED_WARMUP_UPDATES
                    active_opt=(tied_warm_opt if _warm else head_opt); active_params=head_params
                else:
                    active_opt=stage_opts[int(train_stage)]; active_params=stage_param_sets[int(train_stage)]
            active_opt.zero_grad(set_to_none=True)
            if (RPV16_ALWAYS_HEAD and (not global_anchor) and train_stage != JOINT_STAGE):
                head_opt.zero_grad(set_to_none=True)
        t_fetch=time.perf_counter()
        try:
            ids,labels,srcs=stream.batch(args.batch,args.seq)
        except PrefetchAborted:
            # Stop signal while starved for data at an update boundary; nothing was consumed.
            _boundary_service(step-1)
            return
        data_wait_s=time.perf_counter()-t_fetch
        rpv_exact_head_receipt = None
        if tel2 and step == start_step:
            _emit({"event":"stream_ready","ts":time.time(),"step":step,"stream_ff_s":data_wait_s,"prefetch":not args.no_prefetch,"proc_uptime_s":time.time()-_PROC_T0})
        if lazy:
            # CUDA events time the step on the GPU timeline without blocking the host.
            ev0=torch.cuda.Event(enable_timing=True); ev0.record()
        else:
            torch.cuda.synchronize()
        s0=time.perf_counter()
        _ah_obj="ar" if (AH is None or global_anchor) else AH.objective_for(optimizer_update_clock,micro_in_update)
        if _ah_obj == "ar":
            loss,_,aux,counts=model(ids,labels,train_stage=train_stage,head_anchor=global_anchor)
        else:
            loss,aux,counts=AH.forward(_ah_obj,model,ids,labels,train_stage,optimizer_update_clock,micro_in_update)
        if not lazy and not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
        (loss / args.grad_accum).backward()
        micro_in_update += 1
        do_update = (micro_in_update >= args.grad_accum) or (step == args.steps)
        grad_norm = None
        if do_update:
            tokens_after = seen + args.batch*args.seq
            if args.warmup_tokens > 0 and tokens_after < args.warmup_tokens:
                lr_now = args.lr * max(1e-4, tokens_after / args.warmup_tokens)
            else:
                prog = max(0.0,min(1.0,(tokens_after-args.warmup_tokens)/max(1,TOTAL_TRAIN_TOKENS-args.warmup_tokens)))
                lr_now = args.lr * (args.min_lr_mult + (1.0-args.min_lr_mult)*0.5*(1.0+math.cos(math.pi*prog)))
            if train_stage == JOINT_STAGE:
                grad_norm = _joint_update(_ah_obj, lr_now)
            else:
                _lr_use = lr_now
                if tied_warm_opt is not None and active_opt is tied_warm_opt:
                    _lr_use = TIED_WARMUP_LR * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, tied_done_py / max(1, TIED_WARMUP_UPDATES)))))
                for pg in active_opt.param_groups: pg["lr"] = _lr_use
                grad_norm=torch.nn.utils.clip_grad_norm_(active_params,(1.0 if (AH is None or _ah_obj == "ar") else AH.nonar_max_norm(train_stage)))
                if AH is not None and _ah_obj == "ar" and not global_anchor: AH.note_ar_grad_norm(train_stage,grad_norm)
                if not lazy and not torch.isfinite(grad_norm): raise RuntimeError("nonfinite grad norm")
                active_opt.step()
                if (RPV16_ALWAYS_HEAD and (not global_anchor) and train_stage != JOINT_STAGE and _ah_obj == "ar"):
                    # V91_HEAD_LR_DYNAMIC: re-read env each AR head step so soft mult changes bind without code redeploy
                    _hlr = float(os.environ.get("AGILLM_RPV16_HEAD_LR_MULT", str(RPV16_HEAD_LR_MULT)))
                    for pg in head_opt.param_groups: pg["lr"] = _lr_use * _hlr
                    _hgn = torch.nn.utils.clip_grad_norm_(head_params, 1.0)
                    if not lazy and not torch.isfinite(_hgn): raise RuntimeError("nonfinite head grad norm")
                    head_opt.step()
                    head_opt.zero_grad(set_to_none=True)
                    if int(train_stage) != 0 and model.embed.weight.grad is not None:
                        _egn = torch.nn.utils.clip_grad_norm_([model.embed.weight], 1.0)
                        if not lazy and not torch.isfinite(_egn): raise RuntimeError("nonfinite embed grad norm")
                        for pg in stage_opts[0].param_groups: pg["lr"] = _lr_use
                        stage_opts[0].step()
                        stage_opts[0].zero_grad(set_to_none=True)
            # 2026-09-23 cowork: exact RPV head-lift DISABLED (quality regression). From its first live use at
            # step 205,833 every stage's AR loss rose steadily (~7.72 -> ~8.7 by step 516k) while AR grad norms
            # grew 0.04 -> 0.2. Each anchor (every 17 updates) it applied a full single-row Newton step to the
            # shared RPV16 head as a rank-1 weight edit; the 16-row guard includes the solved row, so ~99% of
            # lifts were accepted. The head keeps its ordinary AdamW anchor updates. Function kept for replay.
            if AH is not None: AH.after_step(_ah_obj,lr_now,head_opt)
            if not global_anchor and train_stage != JOINT_STAGE:
                model.invalidate_stage(train_stage)
            if tied_warm_opt is not None and active_opt is tied_warm_opt:
                tied_done_py += 1
                model.tied_warmup_done.fill_(tied_done_py)
                if tied_done_py >= TIED_WARMUP_UPDATES:
                    print(json.dumps({"event":"tied_head_warmup_complete","updates":tied_done_py,"step":step}),flush=True)
            active_opt.zero_grad(set_to_none=True)
            optimizer_update_clock += 1
            micro_in_update = 0
            update_number = optimizer_update_clock
            if _HM_TRIM_EVERY > 0 and update_number % _HM_TRIM_EVERY == 0: _hm_trim("periodic", step)  # V43KN
            _due_updates, _due_time = _checkpoint_periodic_due(
                ckpt_state, update_number, args.save_every_updates, _ckpt_every_sec, time.time())
            if _due_updates or _due_time or _fm_first_save_pending:
                if lazy:
                    # Deferred NaN/Inf guard: every earlier step, then this one, must be finite before capture.
                    _flush_train_log(True)
                    if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
                    if not torch.isfinite(grad_norm): raise RuntimeError("nonfinite grad norm")
                _ok_save = _checkpoint(step,seen + args.batch*args.seq,force_sync=bool(_fm_first_save_pending),reason=("master_migration" if _fm_first_save_pending else ("periodic_updates" if _due_updates else "periodic_time")))
                if _fm_first_save_pending:
                    if not _ok_save:
                        raise RuntimeError("Failed first persistent master checkpoint; refusing unverified continuation")
                    _fm_first_save_pending = False
                    _emit({"event":"fp32_master_checkpoint_persisted", "step":step, "banks":sum(isinstance(o, _FP32MasterOptimizer) for o in stage_opts), "master_numel":_FM_STATS["master_numel"]})
                # Rebase both periodic triggers after any attempt, including an I/O failure.
                # This prevents time/update collisions and retains bounded failure retries.
                ckpt_state["last_periodic_attempt_clock"] = update_number
                ckpt_state["last_save_wall"] = time.time()
                if _due_time and not _due_updates:
                    _emit({"event":"checkpoint_time_trigger","ts":time.time(),"step":step,"optimizer_update_clock":optimizer_update_clock,"every_sec":_ckpt_every_sec,"saved":_ok_save})
        if lazy:
            # float64 on device is exact for the f32/bf16 scalars, so the logged values equal v1's float(x).
            gn_dev=grad_norm.detach().double() if grad_norm is not None else torch.zeros((),device=loss.device,dtype=torch.float64)
            scal=torch.stack([loss.detach().double(),aux.detach().double(),gn_dev]).to("cpu",non_blocking=True)
            ev1=torch.cuda.Event(enable_timing=True); ev1.record()
            dt=loss_v=aux_v=gn_v=None
        else:
            torch.cuda.synchronize(); dt=time.perf_counter()-s0
            step_times.append(dt)
            loss_v=float(loss.detach()); aux_v=float(aux.detach()); gn_v=None if grad_norm is None else float(grad_norm.detach())
        seen += args.batch*args.seq
        tel=model.sparse_telemetry()
        rec={"event":"train","step":step,"seen_tokens":seen,"loss":loss_v,"aux":aux_v,"grad_norm":gn_v,"optimizer_step":do_update,"grad_accum":args.grad_accum,"optimizer_update_clock":optimizer_update_clock,"micro_in_update":micro_in_update,"train_stage":train_stage,"global_anchor":global_anchor,"global_anchor_every":args.global_anchor_every,"lr":float(active_opt.param_groups[0]["lr"]),"optimizer":optimizer_name,"step_s":dt,"tok_s":None if dt is None else args.batch*args.seq/dt,"sources":srcs,"sparse":tel,"cuda_alloc":torch.cuda.memory_allocated(),"cuda_reserved":torch.cuda.memory_reserved(),"route_min":min(min(x) for x in counts),"route_max":max(max(x) for x in counts)}
        if AH is not None: rec.update(AH.telemetry(_ah_obj,args.batch*args.seq))
        if RPV16_TIED_HEAD:
            rec["head"] = "tied_full_vocab"
            if global_anchor and tied_warm_opt is not None and active_opt is tied_warm_opt:
                rec["update_mode"] = "tied_head_warmup"; rec["tied_warmup"] = tied_done_py
        rec["mega_dgrad_silu"] = _mds_v38.telemetry()
        rec["v43j"] = dict(v43j_telemetry(), cslt_down=dict(_CSLT_DOWN_STATS))
        rec["groupedpack_frozen"] = _aq_telemetry()
        if train_stage == JOINT_STAGE:
            rec["update_mode"] = "joint"
            rec["optimizer_streamstep"] = _sg_streamstep.telemetry()
            rec["retained_stage_count"] = _BR_LAST_COUNT
            rec["joint_execution"] = "bounded_retained" if _BR_LAST_COUNT else "replay"
            rec["retention_memory_latched_off"] = _BR_MEMORY_LATCH
            rec["fused_silu_bwd"] = _fsb_telemetry()
            rec["fused_masked_wgrad"] = _cmw_telemetry()
            if do_update and "stage_gn" in _joint_last: rec["joint_stage_gn"] = _joint_last["stage_gn"]
            if do_update and "head_gn" in _joint_last: rec["joint_head_gn"] = _joint_last["head_gn"]
        if rpv_exact_head_receipt is not None:
            rec["rpv_exact_headlift"] = rpv_exact_head_receipt
        if tel2:
            # step_wall_s: host wall time since the previous step's record, i.e. everything
            # (data wait, compute, save stall, logging). Summed over steps it is the run's wall time.
            t_now=time.perf_counter()
            rec.update({"ts":time.time(),"data_wait_s":data_wait_s,"step_wall_s":t_now-t_prev,"prefetch_q":stream.last_qsize})
            t_prev=t_now
        if lazy:
            pending.append((rec,scal,ev0,ev1))
            _flush_train_log(len(pending)>=max(1,args.finite_check_every))
        else:
            _finalize_pipeline_telemetry(rec)
            _emit(rec)
        _report_ckpt(ckpt_writer.finish(block=False),ckpt_state, model=model)
        if micro_in_update == 0 and (_STOP["n"] or ckpt_state["retry"]) and _boundary_service(step):
            return
    if lazy: _flush_train_log(True)
    _report_ckpt(ckpt_writer.finish(block=True),ckpt_state, model=model)
    if ckpt_state["retry"]: _boundary_service(args.steps)
    stream.stop_prefetch()
    result={"schema":"agillm.gb10.1pf.canary.v1","ok":True,"steps":args.steps,"seen_tokens":seen,"median_step_s":statistics.median(step_times),"median_tok_s":args.batch*args.seq/statistics.median(step_times),"parameters":params,"target_training_tokens":TOTAL_TRAIN_TOKENS,"verified_sparse_microkernel_tflops":VERIFIED_SPARSE_TFLOPS,"finished_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    Path(args.receipt).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description="AGILLM-GB10-1PF single-file hardware-locked trainer")
    sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("profile")
    d=sub.add_parser("dataset-probe"); d.add_argument("--dataset",default="fineweb-edu",choices=list(HF_SOURCES)+["mix"])
    t=sub.add_parser("train"); t.add_argument("--dataset",default="fineweb-edu",choices=list(HF_SOURCES)+["mix"]); t.add_argument("--batch",type=int,default=1); t.add_argument("--seq",type=int,default=2048); t.add_argument("--steps",type=int,default=1); t.add_argument("--grad-accum",type=int,default=8); t.add_argument("--legacy-resume-grad-accum",type=int,default=8); t.add_argument("--global-anchor-every",type=int,default=64); t.add_argument("--lr",type=float,default=2e-4); t.add_argument("--warmup-tokens",type=int,default=100000000); t.add_argument("--min-lr-mult",type=float,default=0.1); t.add_argument("--save-every-updates",type=int,default=512,help="periodic full checkpoint cadence in optimizer updates; 0 disables periodic saves"); t.add_argument("--ce-chunk",type=int,default=4096); t.add_argument("--force-head-only",action="store_true"); t.add_argument("--save-dir",default="/workspace/agillm-gb10-1pf-checkpoints"); t.add_argument("--resume",default=""); t.add_argument("--seed",type=int,default=42); t.add_argument("--receipt",default="/workspace/agillm_gb10_1pf_canary.json")
    t.add_argument("--objectives",default="ar,sat,nat",help="'ar,sat,nat' (owner contract) or 'ar' (exact v1 reproduction)")
    t.add_argument("--allheads-hot-config",default="",help="hot JSON reloaded on mtime change; default <save-dir>/allheads_hot.json")
    t.add_argument("--dblock-ar-prob",type=float,default=0.60); t.add_argument("--dblock-sat-prob",type=float,default=0.25); t.add_argument("--dblock-nat-prob",type=float,default=0.15)
    t.add_argument("--sat-fixed-share",type=float,default=0.5); t.add_argument("--satvar-block-probs",default="1:0.5,2:0.5")
    t.add_argument("--satvar-lambda",type=float,default=1.0); t.add_argument("--satvar-merge-cost",type=float,default=-1.0,help="<0 -> 0.5*lambda (S default)")
    t.add_argument("--satvar-regret-weight",type=float,default=0.05); t.add_argument("--satvar-log-every",type=int,default=20); t.add_argument("--satvar-regret-max-blocks",type=int,default=4096,help="fused-CE path only: blocks sampled per SAT step for the regret targets")
    t.add_argument("--nat-mask-rate",default="uniform:0.1:1.0",help="uniform:lo:hi per-sequence sampled rate, or a fixed float")
    t.add_argument("--nat-span-mask-prob",type=float,default=0.35); t.add_argument("--nat-suffix-mask-prob",type=float,default=0.20); t.add_argument("--nat-region-partial-prob",type=float,default=0.5)
    t.add_argument("--nat-attn",default="causal",choices=["causal","bidir"],help="bidir = natmaskedhead_v2 kernel call; position-blind on this NoPE network")
    t.add_argument("--allheads-head-grad",default="sat",choices=["none","sat","sat+nat"]); t.add_argument("--allheads-aux-lr-mult",type=float,default=1.0)
    t.add_argument("--allheads-nonar-clip-mult",type=float,default=1.0,help="SAT/NAT stage gradients are clipped to mult x the stage's typical AR grad norm (log-EMA); 0 = v3_3 behaviour (clip 1.0)")
    # v2 full-stack switches. New behaviour is the default; each flag restores v1 for one feature.
    t.add_argument("--no-prefetch","--no-data-prefetch",action="store_true",help="v1: build batches synchronously on the main thread")
    t.add_argument("--prefetch-depth","--data-prefetch-depth",type=int,default=3,help="ready CPU batches held by the producer (bounded queue)")
    t.add_argument("--no-prefetch-pin",action="store_true",help="do not pin prefetched CPU batches")
    t.add_argument("--sync-checkpoint",action="store_true",help="v1: synchronous torch.save at the save boundary")
    t.add_argument("--async-ckpt-reserve-gb",type=float,default=8.0,help="async save needs MemAvailable >= checkpoint size + this, else it falls back to a synchronous save")
    t.add_argument("--no-ckpt-fsync",action="store_true",help="skip fsync on async and signal saves (periodic synchronous saves never fsync, as in v1)")
    t.add_argument("--strict-step-sync",action="store_true",help="v1: per-step cuda synchronize, blocking isfinite checks, immediate log line")
    t.add_argument("--finite-check-every",type=int,default=16,help="max steps a train record and its NaN/Inf check may be deferred (always forced before a checkpoint)")
    t.add_argument("--v1-telemetry",action="store_true",help="v1: omit the added telemetry keys/events")
    t.add_argument("--v1-behavior",action="store_true",help="compatibility bundle: no-prefetch, sync-checkpoint, strict-step-sync, v1-telemetry; signal-save remains mandatory")
    t.add_argument("--gil-switch-interval-ms",type=float,default=0.0,help="experiment knob for the GPU gate: sys.setswitchinterval; 0 leaves the interpreter default")
    t.add_argument("--stagefwd-v1-routing",action="store_true",help="restore v1 nonzero()/index_copy_ expert dispatch (6 host syncs per stage forward)")
    t.add_argument("--stagefwd-v1-active-silu",action="store_true",help="restore v1 F.silu(gate)*up + separate down-proj quant in the active stage")
    t.add_argument("--stagefwd-v1-repack",action="store_true",help="restore v1 destroy/re-create of native sparse contexts on weight-version change")
    # Promotion gate uses reference CE: fused CE currently omits per-row NLL and
    # activates the 4096-block SAT regret sampling branch. Preserve full SAT-var
    # regret training until fused CE can expose every row NLL without sampling.
    t.add_argument("--fused-ce",action="store_true",help="use fused rank-16 CE kernel; default keeps reference CE")
    a=p.parse_args()
    # v42 production invariant: Scott's live targetfix lineage is always AR+SAT-var+NAT.
    # AR-only remains available for isolated A/B save dirs, but can never silently replace production.
    if a.cmd == "train" and Path(a.save_dir).resolve() == Path("/workspace/agillm-gb10-1pf-targetfix-active").resolve():
        _prod_objectives = sorted(x.strip().lower() for x in str(a.objectives).split(",") if x.strip())
        if _prod_objectives == ["ar"]:
            raise SystemExit("production targetfix forbids --objectives ar; use ar,sat,nat (AR-only is test-dir only)")
        if _prod_objectives != ["ar", "nat", "sat"]:
            raise SystemExit("production targetfix requires exactly ar,sat,nat")
    if a.cmd == "train":
        STAGEFWD["syncfree_routing"] = not a.stagefwd_v1_routing
        STAGEFWD["active_fused_silupack"] = not a.stagefwd_v1_active_silu
        STAGEFWD["inplace_repack"] = not a.stagefwd_v1_repack
        print(json.dumps({"event":"stagefwd_config","variant":"v2b_stagefwd",**STAGEFWD}),flush=True)
    if a.cmd=="profile": profile()
    elif a.cmd=="dataset-probe":
        dataset_probe(list(HF_SOURCES) if a.dataset=="mix" else [a.dataset]); sys.stdout.flush(); os._exit(0)
    else:
        if a.v1_behavior:
            a.no_prefetch = a.sync_checkpoint = a.strict_step_sync = a.v1_telemetry = True
        try:
            train(a)
        finally:
            _shutdown_background()  # stop the producer; let an in-flight checkpoint write finish
        sys.stdout.flush(); os._exit(0)


def _install_rpv2x_aux_runtime():
    # Execution-only speedups validated on sm86. Other architectures retain their
    # existing native/emulated implementations and unchanged checkpoint schema.
    if os.environ.get("AGILLM_RPV2X_AUX", "1") == "0":
        print(json.dumps({"event":"rpv2x_aux_disabled","reason":"operator_flag"}),flush=True)
        return
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) != (8,6):
        print(json.dumps({"event":"rpv2x_aux_disabled","reason":"architecture_baseline_retained"}),flush=True)
        return
    import base64 as _b64, hashlib as _hash, io as _io, zipfile as _zip
    import importlib.util as _iu, sys as _sys
    _blob = _b64.b64decode(
        "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82+xbqSmm8hxU5ime/sm6wZrustvsu6VCaybznHj9ATWtXA7jtIJrLvnBusWT2DddzhuFE5g3YY3WLdA0/3lOuvma7qTr7NulqabwHHTcE33+jXWTdd0l15j3VRNtwfHDaem+xC347Bputuvsm6kpjuC40aYplv3KusGa7rH/2bdK+O1+fw3654er81njptKx7PuxSusWzyedT+9wrqF41m3E8eNAk33fm7Hka/pbr7Mulma7hscN4ZruhGXWTdd0/32Euumarq5l1jXqenGcdxk03QvXGTdSE234CLrhmm6XThuBGu6D3A7jisfsO7WC6x7+gPWHc5xo/QD1q13gXWLP2DdH86zbuEHrJt3nnULNF0nx035mu71c6ybpekuPce6wzXd3hw30jXdMG7Hkarp7v6LdZ2a7iiOGzZNt/FfrBup6Z4+y7phmu60s6wbrOkmcdx0JZd17/pY93Qu6xb6WLc0l3UHctwozmXdGtyOozCXdQ/8yboFmm4Wx418Tdf+J+tmabpn/9Dqr6Y79w/WTdd0UzluStV0DY4bTk134+9a/dV0X+O4EanpRv7OumGa7ndntPqr6Xo4blzJYV3HGdY9ncO6V37T6m8O6y7+jXWLc1i3O8dNhTmsa+K4UaDpFv+q1V9NN5PjRpamG/0r6w7XdH8+rdVfTTef40aqppt4mnWdmu7tU1r91XSXn2LdSE03neOmME3XwnEjWNPd94tWf7NZdwzHjdPZrGv7hXVLs1n395Na/c1m3VkcNwqzWdd9knULNN1gjhv5mu7an7X6q+kO4bhpuKb7OMeNdE334Amt/mq64zhuODXdpidY16bp/vmTVn813U84boRpui/8xLrBmm4ljhtXxrHu+h+1+juOdV/luKl0HOvW4rhRPI51j/6g1d9xrDue40aBptvyB9bN13QvHdfqr6a7kOPGcE2323HWTdd0H+S4karpbv9eq7+a7giOm2yabgOOG5Ga7k/HtPqr6U7kuBGs6cYfY90rWax75Tut/max7mKOG6VZrNv9O9YtzmJdE8eNwizWLf5Wq7+abibHTfmabjTHjSxN9+ejWv3VdPM5bqRruolHWTdV0719RKu/mu5yjhs2TTf9COtGaroWjhthmu6+Uq3+arpjOG66MpZ1bRw3To9l3d8Pa/V3LOvO4rhRPJZ13YdZt3As6947pNVfTfcr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"
        # RPV2X_ARCHIVE_MORE
    )
    if _hash.sha256(_blob).hexdigest() != 'fcae7d8ae744ea380118193b22da9f54db34cf579b58575d3a582ee272384912':
        raise RuntimeError("Embedded RPV runtime archive failed integrity check")
    _hashes = {'candidate_aux_only_v1.py': '41467623c961b0434f837708a9469331e3925864241f20e367caf6b89d1eab0b', 'candidate_cached_wq.py': '867f2677f6658fcafa3f7ad43d7c8556270be54a382c43b1dea4df19170cbc70', 'candidate_frozen_silu.py': '15be54917f6c237845dd2f0fa96d8a3702d6947cb4a43bb07b8dff5904616d2d', 'silu_candidate/silu_candidate.py': 'a3faa1e975b4d8c8246c0bcfd4ef04e333012a0344e330fcc8bc8e459d4f47e6', 'silu_candidate/libfused_siluquant_sm86.so': 'b6bd957bc1ca9c87c23b0e6c78156431c815483eb9b045acfc316a859dd436a6'}
    _root = Path(__file__).resolve().parent / "rpv2x_aux_runtime" / 'fcae7d8ae744ea38'
    _root.mkdir(parents=True,exist_ok=True)
    _resolved_root = _root.resolve()
    with _zip.ZipFile(_io.BytesIO(_blob)) as _zf:
        if set(_zf.namelist()) != set(_hashes):
            raise RuntimeError("Unexpected embedded RPV runtime members")
        for _name,_expected in _hashes.items():
            _data=_zf.read(_name)
            if _hash.sha256(_data).hexdigest()!=_expected:
                raise RuntimeError("Embedded RPV runtime member failed integrity check: "+_name)
            _path=_root/_name
            if not _path.resolve().is_relative_to(_resolved_root):
                raise RuntimeError("RPV runtime extraction escaped its cache")
            _path.parent.mkdir(parents=True,exist_ok=True)
            if not _path.exists() or _hash.sha256(_path.read_bytes()).hexdigest()!=_expected:
                _tmp=_path.with_name(_path.name+".tmp."+str(os.getpid()))
                with _tmp.open("xb") as _f:_f.write(_data)
                os.replace(_tmp,_path)
    _spec=_iu.spec_from_file_location("rpv2x_aux_runtime_v1",_root/"candidate_aux_only_v1.py")
    _mod=_iu.module_from_spec(_spec);_sys.modules[_spec.name]=_mod;_spec.loader.exec_module(_mod)
    _mod.install(_sys.modules[__name__])
    print(json.dumps({"event":"rpv2x_aux_runtime_ready","archive_sha256":'fcae7d8ae744ea380118193b22da9f54db34cf579b58575d3a582ee272384912',
                      "architecture":"sm86","gemm":"original_int8","checkpoint_schema_changed":False}),flush=True)

def _install_rpv2x_gateup_runtime():
    if os.environ.get("AGILLM_CSLT_GATEUP", "1") == "0":
        return
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) != (8,6):
        return
    import base64 as _b64, hashlib as _hash, io as _io, zipfile as _zip
    import importlib.util as _iu, sys as _sys
    _blob=_b64.b64decode('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')
    if _hash.sha256(_blob).hexdigest()!='38bec1cf4818514e0be8a36f05212085fee1ad293552bf105b917090ef29f674':
        raise RuntimeError("Gate/up runtime archive integrity failure")
    _root=Path(__file__).resolve().parent/"rpv2x_gateup_runtime"/'38bec1cf4818514e'
    _root.mkdir(parents=True,exist_ok=True)
    _name="candidate_cslt_gateup_only.py"
    _path=_root/_name
    if not _path.resolve().is_relative_to(_root.resolve()):
        raise RuntimeError("Gate/up runtime extraction escaped its cache")
    with _zip.ZipFile(_io.BytesIO(_blob)) as _zf:
        if _zf.namelist()!=[_name]:raise RuntimeError("Unexpected gate/up archive member")
        _data=_zf.read(_name)
    if _hash.sha256(_data).hexdigest()!='4e854ff1403ae13899d2bc4a35210677ab0bcbd88b9e68eee7f48e02cf8ec944':
        raise RuntimeError("Gate/up runtime member integrity failure")
    if not _path.exists() or _hash.sha256(_path.read_bytes()).hexdigest()!='4e854ff1403ae13899d2bc4a35210677ab0bcbd88b9e68eee7f48e02cf8ec944':
        _tmp=_path.with_name(_path.name+".tmp."+str(os.getpid()))
        with _tmp.open("xb") as _f:_f.write(_data)
        os.replace(_tmp,_path)
    _spec=_iu.spec_from_file_location("rpv2x_gateup_runtime_v1",_path)
    _mod=_iu.module_from_spec(_spec);_sys.modules[_spec.name]=_mod;_spec.loader.exec_module(_mod)
    _mod.install(_sys.modules[__name__],max_cache_mib=1536,alg_id=0)
    _module=_sys.modules[__name__]
    _original_emit=_module._emit
    _emit_state={"first":True}
    def _rpv2x_emit(record):
        if record.get("event")=="train" and (_emit_state["first"] or int(record.get("step",0))%16==0):
            _emit_state["first"]=False
            record["rpv2x"]={
                "aux_enabled":bool(getattr(_module,"_cached_wq_candidate_installed",False)),
                "gateup":dict(getattr(_module,"_cslt_gateup_stats",{})),
                "weight_decode":dict(getattr(_module,"_cached_wq_candidate_stats",{})),
                "frozen_silu":dict(getattr(_module,"_frozen_silu_candidate_stats",{})),
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes":torch.cuda.max_memory_reserved()}
        return _original_emit(record)
    _module._emit=_rpv2x_emit
    print(json.dumps({"event":"rpv2x_gateup_runtime_ready","archive_sha256":'38bec1cf4818514e0be8a36f05212085fee1ad293552bf105b917090ef29f674',"eligible_shape":[8192,5120,1280],"down_and_other_gemm":"original","checkpoint_schema_changed":False}),flush=True)



# ============================================================================ cowork-fable v43k (2026-10-04)
# cuSPARSELt for the expert DOWN projection on sm_86, extending the production rpv2x gate/up candidate
# (candidate_cslt_gateup_only.py: paired-4:8 -> Ampere 2:4 via the in-group column permutation [0,2,4,6,1,3,5,7],
# exact BF16 decode of the NVFP4 operands, FP32 alpha epilogue, single BF16 output rounding) to the
# [n=8192 rows, m=1280 out, k=5120 in] shape that still ran the INT8 block-scaled emulation.
# Same numeric class as the live gate/up path (FP32 tensor-core reduction order instead of exact int32 32-wide
# block sums; identical bf16 operands and alpha). Own LRU cache of compressed down weights, invalidated on
# NativeContext.pack/close exactly like the gate/up cache. Env AGILLM_CSLT_DOWN=0 disables it.
_CSLT_DOWN_STATS = {"run_calls": 0, "compress_calls": 0, "cache_hits": 0, "evictions": 0, "resident_bytes": 0,
                    "peak_resident_bytes": 0, "fallback_calls": 0, "installed": False, "error": None}


def _install_cslt_down_runtime():
    if os.environ.get("AGILLM_CSLT_DOWN", "1") == "0":
        print(json.dumps({"event": "cslt_down_disabled", "reason": "operator_flag"}), flush=True)
        return
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) != (8, 6):
        return
    import sys as _sys, weakref as _wr
    from collections import OrderedDict as _OD
    gu = _sys.modules.get("rpv2x_gateup_runtime_v1")
    if gu is None or not getattr(_sys.modules[__name__], "_cslt_gateup_installed", False):
        _CSLT_DOWN_STATS["error"] = "gateup runtime not installed"
        print(json.dumps({"event": "cslt_down_disabled", "reason": "gateup_runtime_absent"}), flush=True)
        return
    mod = _sys.modules[__name__]
    maximum = int(float(os.environ.get("AGILLM_CSLT_DOWN_CACHE_MIB", "800")) * 2 ** 20)
    alg_id = int(os.environ.get("AGILLM_CSLT_DOWN_ALG_ID", "0"))
    cache = _OD()
    prev_pack, prev_run, prev_close = mod.NativeContext.pack, mod.NativeContext.run, mod.NativeContext.close
    st = _CSLT_DOWN_STATS

    def release(key):
        entry = cache.pop(key, None)
        if entry is not None:
            st["resident_bytes"] -= entry[1].numel() * entry[1].element_size()

    def packed(self, weight_codes, weight_scales):
        release(id(self))
        return prev_pack(self, weight_codes, weight_scales)

    def closed(self):
        release(id(self))
        return prev_close(self)

    @torch.no_grad()
    def run(self, packed_x, sf_x, alpha):
        if not (self.emulated and self.k == 5120 and self.m == 1280 and self.n == 8192
                and self._pw is not None and self._sfw is not None
                and tuple(self._pw.shape) == (1280, 2560) and tuple(self._sfw.shape) == (1280, 160)
                and tuple(packed_x.shape) == (8192, 2560) and tuple(sf_x.shape) == (8192, 160)
                and alpha.numel() == 1 and packed_x.is_contiguous() and sf_x.is_contiguous()
                and self._pw.is_contiguous() and self._sfw.is_contiguous()
                and packed_x.dtype == torch.uint8 and sf_x.dtype == torch.float8_e4m3fn):
            st["fallback_calls"] += 1
            return prev_run(self, packed_x, sf_x, alpha)
        key = id(self)
        entry = cache.get(key)
        if entry is not None and entry[0]() is self:
            comp = entry[1]
            cache.move_to_end(key)
            st["cache_hits"] += 1
        else:
            release(key)
            est = self.m * self.k * 2 * 9 // 16
            while cache and st["resident_bytes"] + est > maximum:
                release(next(iter(cache)))
                st["evictions"] += 1
            comp = gu.compress_weight(self._pw, self._sfw, self.k)
            st["compress_calls"] += 1
            size = comp.numel() * comp.element_size()
            if size <= maximum:
                cache[key] = (_wr.ref(self, lambda _u, key=key: release(key)), comp)
                st["resident_bytes"] += size
                st["peak_resident_bytes"] = max(st["peak_resident_bytes"], st["resident_bytes"])
        st["run_calls"] += 1
        return gu.run_sparse(comp, packed_x, sf_x, alpha, self.k, self.m, alg_id)

    mod.NativeContext.pack = packed
    mod.NativeContext.close = closed
    mod.NativeContext.run = run
    st["installed"] = True
    print(json.dumps({"event": "cslt_down_runtime_ready", "shape": [8192, 1280, 5120], "cache_mib": maximum / 2 ** 20,
                      "alg_id": alg_id, "base": "rpv2x_gateup_runtime_v1"}), flush=True)
# ============================================================================ end cowork-fable v43k block

if __name__=="__main__":
    _install_rpv2x_aux_runtime()
    _install_rpv2x_gateup_runtime()
    _install_cslt_down_runtime()
    main()
