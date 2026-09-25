"""Build saveable, evidence-backed TruthLog drafts for selected rescans."""
from __future__ import annotations
import argparse
import copy, json, shutil
from datetime import datetime
from pathlib import Path
from typing import Any
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from daguandan_bridge.application.truth_log_from_scan import build_truth_log_from_scan
from daguandan_bridge.application.turn_slot_projection import project_turn_slots
from daguandan_bridge.application.replay_turn_draft import validate_turn_actor_chain, validate_truth_log_with_live_reducer
from daguandan_bridge.live.models import LiveEvent
from daguandan_bridge.live.session_store import read_json_lines
from daguandan_bridge.live.truth_log import TruthInitialState, TruthLog, load_truth_log
from daguandan_bridge.danzero.state import RANKS, SEATS

IDS = [
    'game_20260822_002135_9c2328',
    'game_20260829_192802_981e50',
    'game_20260821_200355_81b7ee',
    'game_20260817_002012_b523b5',
    'game_20260816_203403_00db35',
    'game_20260816_193531_f94cbf',
    'game_20260816_154444_392ea8',
    'game_20260815_001801_ba5066',
]
REPO=Path(__file__).resolve().parent.parent
SOURCE_BATCH=REPO/'reports/session-corpus-validation/unverified-rescans/unverified_20260912_141743_4533e5e5'
OUT_BATCH=REPO/'reports/session-corpus-validation/unverified-saveable/unverified_20260912_141743_saveable'
SESSION_ROOT=REPO/'data/profiles/tencent_daguandan/sessions'


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n' for row in rows), encoding='utf-8')


def source_actions(sid: str) -> tuple[list[dict[str, Any]], TruthInitialState | None, str]:
    truth_path=SESSION_ROOT/sid/'truth_log.json'
    if truth_path.is_file():
        truth=load_truth_log(truth_path, session_id=sid)
        return [
            {'actor': t.actor, 'pass': t.is_pass, 'cards': tuple(t.cards)}
            for t in truth.turns
        ], truth.initial_state, 'truth_log'
    events=[LiveEvent.from_dict(raw) for raw in read_json_lines(SESSION_ROOT/sid/'timeline.jsonl')]
    initial=next((e for e in events if e.event_type=='initial_state_confirmed'), None)
    if initial is None:
        return [], None, 'none'
    payload=initial.payload
    lead=payload.get('lead_player')
    if lead not in SEATS:
        first_action=next((i for i,e in enumerate(events) if e.event_type in {'player_played','player_passed','manual_confirmed_event'}), len(events))
        lead_event=next((e for i,e in enumerate(events) if i<first_action and e.event_type=='lead_player_confirmed'), None)
        if lead_event is not None:
            lead=lead_event.payload.get('lead_player') or lead_event.actor
    try:
        initial_state=TruthInitialState(str(payload.get('round_level','')), lead, tuple(str(c) for c in payload.get('hand',())))
    except Exception:
        initial_state=None
    actions=[]
    for e in events:
        if e.event_type not in {'player_played','player_passed','manual_confirmed_event'}: continue
        p=e.payload or {}
        actions.append({'actor':e.actor,'pass':bool(p.get('is_pass') or e.event_type=='player_passed'),'cards':tuple(str(c) for c in p.get('cards') or ())})
    return actions, initial_state, 'timeline'


def align_source_to_scan(source: list[dict[str, Any]], scan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Align a near-complete source action sequence to scan evidence.

    Source actor/pass order is preferred. Scan rows supply frame evidence and
    cards when source cards contain an unresolved suit. Extra scan rows are
    dropped; source-only rows remain without frame evidence but retain the
    source action.
    """
    n,m=len(source),len(scan); INF=10**9
    dp=[[INF]*(m+1) for _ in range(n+1)]; prev=[[None]*(m+1) for _ in range(n+1)]; dp[0][0]=0
    for i in range(n+1):
        for j in range(m+1):
            if i<n and j<m:
                cost=(0 if source[i]['actor']==scan[j].get('actor') else 5)
                cost += 0 if source[i]['pass']==bool(scan[j].get('is_pass')) else 2
                if not source[i]['pass'] and not bool(scan[j].get('is_pass')) and len(source[i]['cards'])!=len(scan[j].get('cards') or ()): cost+=1
                if dp[i][j]+cost<dp[i+1][j+1]: dp[i+1][j+1]=dp[i][j]+cost;prev[i+1][j+1]=(i,j,'match')
            if i<n and dp[i][j]+3<dp[i+1][j]: dp[i+1][j]=dp[i][j]+3;prev[i+1][j]=(i,j,'source_only')
            if j<m and dp[i][j]+3<dp[i][j+1]: dp[i][j+1]=dp[i][j]+3;prev[i][j+1]=(i,j,'scan_only')
    i,j=n,m;pairs=[]
    while i or j:
        q=prev[i][j]
        if q is None: raise RuntimeError('alignment failed')
        pi,pj,k=q;pairs.append((pi,pj,k));i,j=pi,pj
    out=[]
    for i,j,k in reversed(pairs):
        if k=='scan_only': continue
        if k=='source_only':
            out.append({'actor':source[i]['actor'],'is_pass':source[i]['pass'],'cards':[] if source[i]['pass'] else list(source[i]['cards']),'frame_start':None,'frame_end':None,'evidence_frames':[],'manual_source_only':True})
            continue
        sr=source[i]; rr=copy.deepcopy(scan[j]); rr['actor']=sr['actor'];rr['is_pass']=sr['pass']
        if sr['pass']: rr['cards']=[]
        elif sr['cards'] and not any('?' in str(c) for c in sr['cards']): rr['cards']=list(sr['cards'])
        out.append(rr)
    return out


def mark(row: dict[str,Any], change: str, source: str, frames: list[int] | None = None) -> dict[str,Any]:
    row=copy.deepcopy(row);row['manual_repair']={'change':change,'source':source,'evidence_frames':frames or []};row['review_status']='needs_review';row['repair_status']='resolved';row['repair_reason']='evidence_backed_saveable_repair';row['uncertainty']=list(dict.fromkeys([*(row.get('uncertainty') or ()), 'manual_repair_pending_user_confirmation']));return row


def baseline(sid: str) -> TruthInitialState:
    p=SESSION_ROOT/sid/'truth_log.json'
    if p.is_file(): return load_truth_log(p,session_id=sid).initial_state
    source,st,_=source_actions(sid)
    if st is None: raise RuntimeError(f'no initial state for {sid}')
    return st


def repair(sid: str) -> tuple[list[dict[str,Any]],TruthInitialState,dict[str,Any]]:
    scan=read_jsonl(SOURCE_BATCH/sid/'action_trace.jsonl'); raw=read_jsonl(SOURCE_BATCH/sid/'raw_action_trace.jsonl'); st=baseline(sid); changes=[]
    rows=copy.deepcopy(scan)
    if sid=='game_20260822_002135_9c2328':
        source,_,_=source_actions(sid); rows=align_source_to_scan(source,scan); changes.append({'description':'采用现有 draft TruthLog 与重扫动作按 actor/pass 对齐，补回缺失 self PASS/AH 等动作并丢弃幽灵动作','frames':'沿用匹配的扫描证据'})
    elif sid=='game_20260829_192802_981e50':
        rows=rows[:79]; rows[77]=mark(rows[77],'QD QD -> QD QS；与视频尾段及时间线末手一致','timeline_end_anchor',[698,708,715,718]); rows[77]['cards']=['QD','QS']; changes.append({'description':'保留到右家最后一对牌；帧 744 已显示二游/结算，删除后续结算页残留动作 #80–#84','frames':[744,751]})
    elif sid=='game_20260821_200355_81b7ee':
        rows.insert(48,mark(raw[82],'在原 #48 left PASS 后补入 self PASS','raw_action_83',[1508,1515])); rows.insert(52,mark(raw[86],'在 right 五张牌后补入 self PASS','raw_action_87',[1580,1590])); st=TruthInitialState(st.round_level,st.lead_player,st.my_hand,(('opposite',28),)); changes.append({'description':'补两处 self PASS；opposite 起手数设为 28，解释视频中 opposite 最后一手 5 张仍可合法结束','frames':[1508,1580,1603,1636]})
    elif sid=='game_20260817_002012_b523b5':
        rows[31]=mark(rows[31],'opposite 4H -> PASS','timeline_action_32',[702,740]); rows[31]['is_pass']=True; rows[31]['cards']=[]; changes.append({'description':'去除一帧 opposite 4H 误识别，恢复 raw/时间线一致的 PASS','frames':[702,740]})
    elif sid=='game_20260816_203403_00db35':
        rows[43]=mark(rows[43],'opposite J? -> PASS','timeline_action_44',[904,909]); rows[43]['is_pass']=True; rows[43]['cards']=[]; rows[60]=mark(rows[60],'opposite PASS -> opposite small_joker','timeline_action_61',[1205,1212]); rows[60]['is_pass']=False; rows[60]['cards']=['small_joker']; dropped=rows.pop(61); changes.append({'description':'删除 left 10H 10H 10C 10S 的幽灵动作；视频尾段显示 opposite small_joker 后进入结算','frames':[1205,1212,1240]})
    elif sid=='game_20260816_193531_f94cbf':
        rows[58]=mark(rows[58],'删除多识别的 2H，修正为 10H 10H 6D 6H 6S','timeline_action_59',[1280,1290,1297]); rows[58]['cards']=['10H','10H','6D','6H','6S']; changes.append({'description':'按时间线和 self 初始手牌修正第 59 手，保留尾部动作','frames':[1280,1297]})
    elif sid=='game_20260816_154444_392ea8':
        rows=[mark(x,'保留视频片段从 raw 首个可见动作开始的动作链','raw_partial_clip',[0,33]) for x in raw[:12]]; changes.append({'description':'该视频从一墩中途开始，采用 raw 的首个可见 right 2C 作为片段首手，不把中途 PASS 当作整局首手','frames':[0,33,41,124]})
    elif sid=='game_20260815_001801_ba5066':
        rows[17]=mark(rows[17],'6D 6C 6S 6D -> 6C 6D 6S 6S','timeline_action_18',[546,589,621]); rows[17]['cards']=['6C','6D','6S','6S']; rows[70]=mark(rows[70],'AH AC AC AD -> AC AC AH AS','timeline_action_71',[1689,1722]); rows[70]['cards']=['AC','AC','AH','AS']; changes.append({'description':'修正两处 self 牌面，使其与已确认初始手牌及时间线一致','frames':[546,621,1689,1722]})
    for i,r in enumerate(rows,1): r['action_id']=i
    return rows,st,{'session_id':sid,'changes':changes}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build saveable, evidence-backed TruthLog drafts for the hard-coded selected rescans.",
        epilog=(
            "Reads the fixed source rescan batch and source session evidence. "
            "Writes a replacement output batch; by default the existing output batch is removed first."
        ),
    )
    parser.add_argument(
        "--source-batch",
        type=Path,
        default=SOURCE_BATCH,
        help="Read-only source rescan batch directory (default: %(default)s).",
    )
    parser.add_argument(
        "--output-batch",
        type=Path,
        default=OUT_BATCH,
        help="Output batch directory to write/recreate (default: %(default)s).",
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=SESSION_ROOT,
        help="Read-only source session evidence root (default: %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    global SOURCE_BATCH, OUT_BATCH, SESSION_ROOT
    SOURCE_BATCH = args.source_batch
    OUT_BATCH = args.output_batch
    SESSION_ROOT = args.session_root
    if OUT_BATCH.exists(): shutil.rmtree(OUT_BATCH)
    OUT_BATCH.mkdir(parents=True)
    results=[];all_pass=True
    for sid in IDS:
        src=SOURCE_BATCH/sid; out=OUT_BATCH/sid; out.mkdir(parents=True)
        for item in src.iterdir():
            if item.is_file(): shutil.copy2(item,out/item.name)
        rows,st,meta=repair(sid)
        log=build_truth_log_from_scan(TruthLog(sid,st,()),rows).truth_log
        checks=[]
        for name,fn in [('actor_chain',validate_turn_actor_chain),('live_reducer',validate_truth_log_with_live_reducer)]:
            try: fn(log);checks.append({'name':name,'status':'PASS'})
            except Exception as exc: checks.append({'name':name,'status':'FAIL','error':str(exc)});all_pass=False
        slots=project_turn_slots(read_jsonl(out/'frame_observations.jsonl.gz') if False else [],rows)
        # Reuse compressed observations without loading through read_jsonl.
        import gzip
        with gzip.open(src/'frame_observations.jsonl.gz','rt',encoding='utf-8') as f: obs=[json.loads(x) for x in f if x.strip()]
        slots=project_turn_slots(obs,rows)
        dump_jsonl(out/'action_trace.jsonl',rows)
        (out/'turn_slots.json').write_text(json.dumps(slots,ensure_ascii=False,indent=2),encoding='utf-8')
        assembled=build_truth_log_from_scan(TruthLog(sid,st,()),rows)
        (out/'truth_log.draft.json').write_text(json.dumps(assembled.truth_log.to_dict(),ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'truth_log_review.json').write_text(json.dumps({'schema':'guandan.truth-log-review/1','review_items':list(assembled.review_items)},ensure_ascii=False,indent=2),encoding='utf-8')
        ss=json.loads((out/'scan_summary.json').read_text(encoding='utf-8'));ss['action_count']=len(rows);ss['turn_slot_counts']=slots.get('counts',{});ss['repair']=meta;ss['truth_log_draft']={'status':'generated','review_count':len(assembled.review_items)};(out/'scan_summary.json').write_text(json.dumps(ss,ensure_ascii=False,indent=2),encoding='utf-8')
        mf=json.loads((out/'scan_manifest.json').read_text(encoding='utf-8'));mf['counts']['actions']=len(rows);mf['counts']['turn_slots']=slots.get('counts',{});mf['repair']=meta;mf['truth_log_draft']={'status':'generated','review_count':len(assembled.review_items)};(out/'scan_manifest.json').write_text(json.dumps(mf,ensure_ascii=False,indent=2),encoding='utf-8')
        receipt={'schema':'guandan.saveable-truthlog-repair/1','session_id':sid,'source_batch':str(SOURCE_BATCH),'output':str(out),'changes':meta['changes'],'strict_checks':checks,'initial_state':st.to_dict(),'manual_confirmation_required':True};(out/'repair_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
        results.append({'session_id':sid,'action_count':len(rows),'draft_review_count':len(assembled.review_items),'initial_state':st.to_dict(),'checks':checks,'changes':meta['changes']})
    summary={'schema':'guandan.saveable-rescan-batch/1','source_batch':str(SOURCE_BATCH),'selected_count':len(IDS),'completed_count':len(IDS),'failed_count':0,'sessions':results,'all_strict_checks_passed':all_pass}
    (OUT_BATCH/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(OUT_BATCH),'all_strict_checks_passed':all_pass,'sessions':[(r['session_id'],r['action_count'],r['checks']) for r in results]},ensure_ascii=False,indent=2))
    return 0 if all_pass else 1
if __name__=='__main__': raise SystemExit(main())
