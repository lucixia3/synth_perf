set -e
ISLES=$1
ATLES=$2
OUT=${3:-sortida}
cd "$(dirname "$0")"
D=../dades
cat $D/train.txt $D/val.txt > $D/tots.txt
python -u registra_mni.py --isles "$ISLES" --atles "$ATLES" --casos $D/tots.txt --out "$OUT/mni"
python -u registra_extra.py --isles "$ISLES" --casos $D/tots.txt --mni_dir "$OUT/mni"
python -u inicialitza_atles.py --mni_dir "$OUT/mni" --vasos $D/vas_ocluit_i_territori.json --train $D/train.txt --out "$OUT/taula_inicial.npy"
V="--mni_dir $OUT/mni --vasos $D/vas_ocluit_i_territori.json --train $D/train.txt --val $D/val.txt --taula $OUT/taula_inicial.npy --workers 3 --reprendre"
python -u entrena.py $V --entrada cta --epoques 40 --out "$OUT/exp/cta" &
python -u entrena.py $V --entrada cta --aug_fort --dropout 0.2 --wd 1e-4 --pes_lesio 3 --lambda_dice 0.5 --criteri dice --epoques 30 --out "$OUT/exp/cta_lesio" &
wait
python -u avalua.py --exp "$OUT/exp" --isles "$ISLES" --vasos $D/vas_ocluit_i_territori.json --train $D/train.txt --val $D/val.txt --out "$OUT/exp/avaluacio"
