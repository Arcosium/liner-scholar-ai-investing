#!/bin/bash
# 별도 기간(2026-03~05) LightGBM v2 평가. HYFE work/wf.sh 의 v2 구성(xs+small 4h·60·84, xs+reg 4h·60·42, xs+small 4h·120·42)을
# 같은 옵션으로 한 분기만 학습: 검증 2025-12~02, 시험 2026-03~05, 종목군 = 2025-12 이전 12개월 거래대금 상위 200(CNN 별도 기간과 같음).
set -euo pipefail; cd /home/arcosium/projects/HYFE_QTPA; export HYFE_BARS=work/bars_hold
O=/home/arcosium/projects/라이너스콜라_공모전/_workspace/final_20261029/gbm; mkdir -p $O; Q=2026-03,2026-03
python3 -m hyfe.pilot_gbm --liq_months 12 --top 200 --quarters $Q --stride 1 --label relbin --xs --preset small --save_pred --configs 4h:60:84,4h:120:42 --out $O/hold_xss_relbin.csv 2>&1 | grep -vE "Warning|warn"
python3 -m hyfe.pilot_gbm --liq_months 12 --top 200 --quarters $Q --stride 1 --label relbin --xs --preset reg --save_pred --configs 4h:60:42 --out $O/hold_xsr_relbin.csv 2>&1 | grep -vE "Warning|warn"
python3 -m hyfe.zavg --norm_os --out $O/ens_v2hold_4h_s0_pred.npz $O/hold_xss_relbin_4h_W60_H84_s0_pred.npz $O/hold_xsr_relbin_4h_W60_H42_s0_pred.npz $O/hold_xss_relbin_4h_W120_H42_s0_pred.npz 2>&1 | grep -v Warn
echo GBM_HOLD_DONE
