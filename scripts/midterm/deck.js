// Mid-term presentation (pptx) of ISCO, built from runs/midterm_reports/{deck/deck_data.json, figures, deck/stills}.
//   NODE_PATH=runs/midterm_reports/deck/node_modules node scripts/midterm/deck.js [out.pptx]
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const WORK = path.join(ROOT, "runs", "midterm_reports");
const FIG = path.join(WORK, "figures", "light");
const STILL = path.join(WORK, "deck", "stills");
const D = JSON.parse(fs.readFileSync(path.join(WORK, "deck", "deck_data.json"), "utf8"));
const OUT = process.argv[2] || path.join(WORK, "ISCO_중간발표.pptx");
const VIDEO = process.env.ISCO_DECK_VIDEO || path.join(WORK, "ISCO_중간발표_데모영상_압축.mp4");
const SKILL = process.env.PPTX_SKILL_DIR;

const THEME = {
  name: "ISCO Midterm", headFontFace: "Malgun Gothic", bodyFontFace: "Malgun Gothic",
  colors: { dk1: "1E2328", lt1: "FFFFFF", dk2: "4A5560", lt2: "F2F4F6", accent1: "E8A317", accent2: "2A78D6",
            accent3: "1BAF7A", accent4: "EB6834", accent5: "E87BA4", accent6: "0F8A3C", hlink: "2A78D6",
            folHlink: "6250D6" },
};
const HEX = { ink: "1E2328", ink2: "4A5560", muted: "898781", grid: "E1E0D9", axis: "C3C2B7",
              series: ["2A78D6", "EB6834", "1BAF7A", "EDA100", "E87BA4", "008300"] };
const TYPES_KO = ["눌림", "돌출", "구멍", "절단", "균열", "휨"];

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.33 x 7.5 in
pres.author = "팀 강화실패 (박정규, 김상준)";
pres.company = "고려대학교 산학캡스톤디자인";
pres.title = "ISCO 중간 점검";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE", background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.0, w: 6.6, h: 1.6, fontSize: 48, bold: true,
                                color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.8, w: 6.6, h: 2.6, fontSize: 18,
                                color: C.background2, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION", background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.9, y: 2.6, w: 11.5, h: 1.2, fontSize: 40, bold: true,
                                color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.9, y: 3.95, w: 11.5, h: 1.6, fontSize: 18,
                                color: C.background2, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.32, w: 10.6, h: 0.8, fontSize: 30, bold: true,
                                color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "ISCO 중간 점검 · 2026년 10월 26일 기준", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontSize: 10,
                                                                     color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 6.98, w: 0.6, h: 0.3, fontSize: 10, color: C.text2, align: "right" },
});

// ------------------------------------------------------------------ helpers
let section = null;
function newSlide(master, title) {
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  if (title) s.addText(title, { placeholder: "title" });
  return s;
}
function addSection(title) { section = title; pres.addSection({ title }); }
function text(s, t, o) { s.addText(t, Object.assign({ isTextBox: true, margin: 0, fontSize: 15, color: C.text1, valign: "top" }, o)); }
function bullets(s, items, o) {
  const runs = items.map((it, i) => {
    const run = typeof it === "string" ? { text: it, options: {} } : it;
    run.options = Object.assign({ bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 }, run.options);
    return run;
  });
  s.addText(runs, Object.assign({ isTextBox: true, margin: 0, fontSize: 15, color: C.text1, valign: "top" }, o));
}
function card(s, x, y, w, h, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: opts.fill || C.background2 },
    line: { color: opts.line || C.background2, width: 1 }, objectName: opts.name || "card" });
}
function badge(s, label, x, y, kind) {
  const fill = kind === "yes" ? C.accent6 : kind === "no" ? C.accent4 : C.text2;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 1.35, h: 0.42, rectRadius: 0.21, fill: { color: fill },
    line: { color: fill, width: 0 }, objectName: "verdict badge" });
  s.addText(label, { isTextBox: true, x, y, w: 1.35, h: 0.42, margin: 0, fontSize: 14, bold: true, color: C.background1,
    align: "center", valign: "middle" });
}
function numberDot(s, n, x, y, d = 0.48) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 },
    objectName: "number" });
  s.addText(String(n), { isTextBox: true, x, y, w: d, h: d, margin: 0, fontSize: 16, bold: true, color: C.text1,
    align: "center", valign: "middle" });
}
function stat(s, x, y, w, big, label, color, size = 44) {
  text(s, big, { x, y, w, h: 0.9, fontSize: size, bold: true, color: color || C.accent2, valign: "bottom", fit: "none" });
  text(s, label, { x, y: y + 0.95, w, h: 0.75, fontSize: 14, color: C.text2 });
}
function img(s, file, x, y, w, h, alt) {
  s.addImage({ path: file, x, y, w, h, altText: alt || path.basename(file), sizing: { type: "contain", w, h } });
}
function pct(v) { return `${Math.round(v * 100)}%`; }
const AX = { catAxisLabelColor: HEX.ink2, valAxisLabelColor: HEX.ink2, catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
             catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: HEX.grid, size: 0.75 },
             catGridLine: { style: "none" }, legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: HEX.ink2,
             titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: HEX.ink, dataLabelFontFace: "+mn-lt",
             dataLabelFontSize: 10, dataLabelColor: HEX.ink2, catAxisLineShow: true, valAxisLineShow: false };

// ------------------------------------------------------------------ 1. title
{
  const s = pres.addSlide({ masterName: "TITLE" });
  s.addText("ISCO 중간 점검", { placeholder: "title" });
  s.addText([
    { text: "제조 공정 환경에서 불량 검출을 위한 제품별 공정 최적화 세팅 시뮬레이터", options: { breakLine: true, bold: true } },
    { text: "Inspection Setup Configuration Optimizer", options: { breakLine: true, color: C.accent1 } },
    { text: " ", options: { breakLine: true } },
    { text: "2026-2 산학캡스톤디자인 · 팀 강화실패", options: { breakLine: true } },
    { text: "박정규(팀장, 컴퓨터학과) · 김상준(데이터과학과)", options: { breakLine: true } },
    { text: "2026년 10월 26일 기준", options: {} },
  ], { placeholder: "body" });
  img(s, path.join(STILL, "line.jpg"), 7.75, 0.75, 5.1, 2.87, "Isaac Sim 컨베이어 디지털 트윈");
  img(s, path.join(STILL, "part_tless_29.jpg"), 7.75, 3.85, 5.1, 2.87, "검사 지점을 지나는 부품과 카메라");
  s.addNotes("ISCO 중간 점검 발표입니다. 부품의 3D 모델과 결함 종류를 넣으면 가상 컨베이어 공정 안에서 카메라 배치, 회전대, 벨트 속도를 직접 시험해 보고, 정상품 오판정 1%에서 놓치는 불량 비율과 비용을 함께 따져 가장 좋은 검사 설정을 찾는 시뮬레이터입니다. 오른쪽은 실제로 만든 Isaac Sim 공정 화면입니다.");
}

// ------------------------------------------------------------------ 2. summary
{
  const s = newSlide("CONTENT", "한 장 요약");
  stat(s, 0.6, 1.35, 2.9, "9 / 9", "시뮬레이터·계산 방법\n점검 통과", C.accent2);
  stat(s, 3.7, 1.35, 2.9, "6종", "제품별 권장 검사 설정\n(카메라 1–3대)", C.accent2);
  stat(s, 6.8, 1.35, 3.0, "1 · 3 · 1", "처음 가설 다섯 개:\n지지 · 기준 미달 · 진행 중", C.accent4);
  stat(s, 10.0, 1.35, 2.8, "8–42%p", "사이클 제한을 지키는 대가\n(미검률 증가)", C.accent4);
  card(s, 0.6, 3.75, 5.95, 2.95);
  card(s, 6.8, 3.75, 5.95, 2.95);
  numberDot(s, 1, 0.85, 4.0);
  numberDot(s, 2, 7.05, 4.0);
  text(s, "결함이 보이는 것과 잡히는 것은 다르다", { x: 1.5, y: 4.02, w: 4.9, h: 0.45, fontSize: 18, bold: true });
  text(s, "가시성이 높은 설정에서도 결함 36개 중 32–34개가 이미 보이는데 57–85%를 놓쳤다. 그래서 가시성(커버리지)이 아니라 검출기의 실제 미검률로 직접 탐색해야 한다.",
       { x: 1.0, y: 4.65, w: 5.3, h: 1.9, fontSize: 15, color: C.text2 });
  text(s, "공정의 사이클 시간 제한을 넣어야 한다", { x: 7.7, y: 4.02, w: 4.9, h: 0.45, fontSize: 18, bold: true });
  text(s, "정지 촬영으로 찾은 최적 설정은 3개 제품 모두 컨베이어 라인의 1.5초 제한을 넘었다. 제한을 지키면 권장 설정이 달라진다(움직임 자체의 영향은 작았다).",
       { x: 7.2, y: 4.65, w: 5.3, h: 1.9, fontSize: 15, color: C.text2 });
  s.addNotes("네 숫자로 요약하면: 시뮬레이터 점검 9개 모두 통과, 6개 제품의 권장 설정을 찾았고, 처음 세운 가설 다섯 중 하나가 지지, 셋은 미리 정한 기준에 못 미쳤고, 하나는 오류를 고쳐 다시 돌리는 중입니다. 데이터가 일관되게 말하는 결론은 두 가지입니다. 첫째, 보이는 것과 잡히는 것은 다르므로 검출기 성능으로 직접 찾아야 하고, 둘째, 사이클 시간 제한을 탐색에 넣어야 실제 라인에서 쓸 수 있는 설정이 나옵니다.");
}

// ------------------------------------------------------------------ section 1
addSection("무엇을 하려 했나");
{
  const s = newSlide("SECTION", "1. 무엇을 하려 했나");
  s.addText("계획서와 발표 자료에서 세운 문제, 목표, 가설", { placeholder: "body" });
}
{
  const s = newSlide("CONTENT", "문제: 3D 검사 라인의 설정은 아직 경험에 의존한다");
  bullets(s, [
    "싱크마크·휨·눌림·미세 단차처럼 모양만 달라지는 결함은 2D 카메라로 잘 안 잡혀 3D 검사가 늘고 있다",
    "그러나 카메라 수·각도·거리, 회전 여부, 컨베이어 속도는 경험으로 정한다",
    "결함 면이 어느 카메라에도 안 보이면, 검출 모델이 아무리 좋아도 놓친다",
    "품목이 자주 바뀌는 라인에서는 바뀔 때마다 다시 설계해야 한다",
    "기존 도구는 정지 부품의 표면 커버리지·측정 품질을 기준으로 하거나(검사 계획 도구), 공정 흐름만 본다(공정 디지털 트윈)",
  ], { x: 0.6, y: 1.4, w: 6.3, h: 4.8, fontSize: 16 });
  const q = ["카메라를 몇 대 둘 것인가", "어느 각도·거리에 둘 것인가", "부품을 돌려 다시 찍을 것인가", "컨베이어를 얼마나 빨리 돌릴 것인가"];
  q.forEach((t, i) => {
    const x = 7.4 + (i % 2) * 2.75, y = 1.45 + Math.floor(i / 2) * 1.75;
    card(s, x, y, 2.55, 1.5);
    numberDot(s, i + 1, x + 0.2, y + 0.2, 0.42);
    text(s, t, { x: x + 0.2, y: y + 0.75, w: 2.2, h: 0.65, fontSize: 15, bold: true });
  });
  card(s, 7.4, 5.15, 5.3, 1.25, { fill: C.text1, line: C.text1 });
  text(s, "빈칸: 실제 공정 흐름 안에서, 실제 검출 성능을 기준으로 검사 설정을 찾는 도구", { x: 7.65, y: 5.32, w: 4.85, h: 0.95,
       fontSize: 15, bold: true, color: C.background1, valign: "middle" });
  s.addNotes("라인 설계자가 정해야 하는 네 가지를 오른쪽에 적었습니다. 기존 도구는 정지 부품의 커버리지나 측정 품질을 기준으로 하거나 공정 흐름만 보기 때문에, 실제 공정 흐름 안에서 실제 검출 성능을 기준으로 설정을 찾는 도구가 비어 있습니다.");
}
{
  const s = newSlide("CONTENT", "목표: 부품이 들어오면 최적 검사 공정을 알려 준다");
  const cols = [
    ["입력", ["부품 3D 모델 (설계도 또는 실제 스캔)", "결함 종류 목록", "제약: 카메라 수·처리 시간"]],
    ["가상 공정 탐색", ["Isaac Sim 컨베이어 공정에서 설정을 바꿔 가며 시뮬레이션", "검출 성능과 비용을 측정"]],
    ["출력", ["권장 공정 설정", "성능–비용 절충 곡선", "결함 종류별 가시성·검출률"]],
  ];
  cols.forEach(([h, items], i) => {
    const x = 0.6 + i * 4.15;
    card(s, x, 1.35, 3.85, 2.55);
    text(s, h, { x: x + 0.25, y: 1.5, w: 3.4, h: 0.45, fontSize: 19, bold: true, color: C.accent2 });
    bullets(s, items, { x: x + 0.25, y: 2.05, w: 3.4, h: 1.75, fontSize: 15 });
    if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 3.88, y: 2.45, w: 0.24, h: 0.36, fill: { color: C.text2 },
                                                    line: { color: C.text2, width: 0 } });
  });
  card(s, 0.6, 4.2, 6.0, 2.5, { fill: C.background1, line: C.accent2 });
  text(s, "성능 = 과검률 1%에서의 미검률", { x: 0.85, y: 4.35, w: 5.6, h: 0.45, fontSize: 18, bold: true });
  text(s, "정상품 100개 중 1개만 불량으로 잘못 걸리도록 판정 기준선을 정한 뒤, 그 기준으로 결함 부품 중 몇 %를 놓치는지 센다. 낮을수록 좋다. 평균 점수가 아니라 \"쓸 만한 오판정 수준에서 몇 개를 놓치는가\"를 본다.",
       { x: 0.85, y: 4.9, w: 5.55, h: 1.7, fontSize: 14, color: C.text2 });
  card(s, 6.8, 4.2, 5.95, 2.5, { fill: C.background1, line: C.accent4 });
  text(s, "비용 = 카메라 수 + 사이클 시간", { x: 7.05, y: 4.35, w: 5.6, h: 0.45, fontSize: 18, bold: true });
  text(s, "사이클 시간 = 부품 간격 0.3 m ÷ 벨트 속도 + 회전대 시간. 벨트 0.1/0.3/0.6 m/s → 3.0/1.0/0.5초, 회전대 90° 0.9초 · 180° 1.1초 · 90°×3 1.7초. 제약 안에서 가장 덜 놓치는 설정을 권장한다.",
       { x: 7.05, y: 4.9, w: 5.5, h: 1.7, fontSize: 14, color: C.text2 });
  s.addNotes("계획서의 목표 그대로입니다. 성능 지표는 과검률 1%에서의 미검률, 비용은 카메라 수와 사이클 시간입니다. 이 목표와 지표는 지금까지 바뀌지 않았습니다.");
}
{
  const s = newSlide("CONTENT", "처음 세운 가설 다섯 가지");
  const H = [
    ["최적 설정은 부품마다 다르다", "다른 부품의 최적 설정을 썼을 때 미검률 손실"],
    ["가시성(결함 표면이 보이는 비율)은 검출 성능을 대신할 수 있다", "가시성과 실제 미검률의 순위상관 → 빠른 1단계 거르기의 근거"],
    ["시뮬레이션 설정 순위가 실제 결함 형상에서도 유지된다", "실제 결함 스캔으로 만든 부품에서 같은 설정들의 순위 비교"],
    ["공정(속도·촬영 시점·사이클)을 반영하면 권장 설정이 달라진다", "정지 촬영 가정 최적 vs 인라인 공정 최적"],
    ["카메라를 더하면 오히려 나빠질 수 있다 (검출기 고유 성질)", "외부 조사에서 추가: 가시성 기준 탐욕적 추가 vs 검출 성능 기준 탐색"],
  ];
  H.forEach(([h, how], i) => {
    const y = 1.35 + i * 0.98;
    numberDot(s, i + 1, 0.6, y + 0.12);
    text(s, h, { x: 1.3, y: y + 0.05, w: 7.4, h: 0.45, fontSize: 16.5, bold: true });
    text(s, how, { x: 1.3, y: y + 0.5, w: 7.4, h: 0.38, fontSize: 13, color: C.text2 });
  });
  card(s, 8.9, 1.35, 3.85, 4.75);
  text(s, "계획서의 예상", { x: 9.15, y: 1.55, w: 3.4, h: 0.45, fontSize: 18, bold: true, color: C.accent2 });
  bullets(s, ["단순한 원기둥 핀(dowel): 카메라 소수, 회전 불필요", "나사산·홈이 많은 케이블 글랜드: 다각도 배치, 회전 필요",
              "실물 앵커: MVTec 3D-AD(센서 차이), Real3D-AD(순위 유지)"], { x: 9.15, y: 2.15, w: 3.4, h: 3.8, fontSize: 15 });
  s.addNotes("발표 때 제시한 네 질문이 가설 1에서 4이고, 5번은 외부 조사 단계에서 덧붙인 가설입니다. 모든 판정 기준은 실험 전에 문서로 고정했습니다.");
}
{
  const s = newSlide("CONTENT", "출발점: 팀원이 만든 시제품 (2026년 9월)");
  card(s, 0.6, 1.35, 5.9, 3.6);
  text(s, "된 것", { x: 0.85, y: 1.5, w: 5.4, h: 0.45, fontSize: 18, bold: true, color: C.accent6 });
  bullets(s, ["엔비디아 공식 컨베이어, 벨트가 부품을 밀어 움직임", "공개 부품 5종을 연속으로 흘려보내기, 영상 저장",
              "카메라 3대(120° 간격) 동시 촬영 → 3D 점 합치기", "정답 모델 대비 표면 덮음 62.4%, 정확도 67.6%"],
          { x: 0.85, y: 2.05, w: 5.5, h: 2.8, fontSize: 15 });
  card(s, 6.85, 1.35, 5.9, 3.6);
  text(s, "안 된 것", { x: 7.1, y: 1.5, w: 5.4, h: 0.45, fontSize: 18, bold: true, color: C.accent4 });
  bullets(s, ["결함 개념 없음, 검출기 없음", "설정 탐색 없음 (카메라 배치가 코드에 고정)", "실제 데이터 연결·수치 안정성 검증 없음",
              "939줄짜리 스크립트 하나"], { x: 7.1, y: 2.05, w: 5.5, h: 2.8, fontSize: 15 });
  card(s, 0.6, 5.2, 12.15, 1.5, { fill: C.text1, line: C.text1 });
  text(s, "50% → 100%", { x: 0.9, y: 5.35, w: 3.6, h: 1.2, fontSize: 30, bold: true, color: C.accent1, valign: "middle" });
  text(s, "잡음 없는 카메라인데 정확도가 낮은 것을 추적: 부품 위치를 읽은 뒤 벨트가 부품을 50 mm 더 옮긴 상태에서 찍고 있었다. 촬영 직전 부품을 고정하도록 고쳐 정확도(0.2 mm 기준)가 50%에서 100%로.",
       { x: 4.6, y: 5.35, w: 7.9, h: 1.2, fontSize: 15, color: C.background1, valign: "middle" });
  s.addNotes("시작할 때는 컨베이어와 촬영까지만 있었고 결함, 검출기, 탐색은 없었습니다. 시제품의 숨은 버그를 찾아 고친 것이 첫 성과였습니다.");
}

// ------------------------------------------------------------------ section 2
addSection("무엇을 만들었나");
{
  const s = newSlide("SECTION", "2. 무엇을 만들었나");
  s.addText("계획서의 다섯 부분이 모두 동작하는 시스템", { placeholder: "body" });
}
{
  const s = newSlide("CONTENT", "시스템 구조: 다섯 부분이 모두 동작한다");
  img(s, path.join(FIG, "system_flow.png"), 0.6, 1.3, 12.1, 5.05, "시스템 흐름도");
  text(s, "명령 한 줄(isco run <부품> --constraint \"n<=3,T<=1.5\")로 부품 투입부터 권장 설정·리포트까지 실행된다.",
       { x: 0.6, y: 6.4, w: 12.1, h: 0.45, fontSize: 14, color: C.text2 });
  s.addNotes("입력, 데이터 생성기, 가상 공정, 검출 파이프라인, 설정 최적화기, 결과 리포트까지 계획서의 다섯 부분이 모두 동작합니다.");
}
{
  const s = newSlide("CONTENT", "가상 공정 환경과 바꿔 보는 설정");
  img(s, path.join(STILL, "slots.jpg"), 0.6, 1.35, 4.2, 2.36, "카메라 후보 자리 96곳");
  text(s, "카메라 후보 자리 96곳 (파랑 0.25 m, 주황 0.40 m)", { x: 0.6, y: 3.78, w: 4.2, h: 0.35, fontSize: 12, color: C.text2 });
  img(s, path.join(STILL, "turntable.jpg"), 0.6, 4.25, 4.2, 2.36, "회전대");
  text(s, "회전대: 멈추고 돌려 가며 촬영", { x: 0.6, y: 6.65, w: 4.2, h: 0.3, fontSize: 12, color: C.text2 });
  const rows = [
    [{ text: "설정", options: { bold: true, color: C.background1, fill: { color: C.text1 } } },
     { text: "범위", options: { bold: true, color: C.background1, fill: { color: C.text1 } } }],
    ["카메라 수", "1–6대"],
    ["카메라 위치", "방위 12 × 고도 4(20/40/60/80°) × 거리 2(0.25/0.40 m) = 96곳, 서로 30° 이상"],
    ["회전대", "없음 / 90° 한 번 / 180° 한 번 / 90°씩 세 번"],
    ["벨트 속도", "0.1 / 0.3 / 0.6 m/s"],
    ["촬영 방식", "모든 카메라 동시 / 순서대로"],
    ["센서 잡음 (기본)", "0.25 m에서 0.15 mm, 거리 제곱에 비례, 비스듬한 면 측정 실패, 경계 튐 제거"],
    ["벨트 움직임", "촬영 신호 지연 → 위치 오차(0.6 m/s에서 3 mm), 노출 중 번짐 1.2 mm"],
    ["회전대 오차", "다시 놓일 때 0.5 mm · 1°"],
  ];
  s.addTable(rows, { x: 5.1, y: 1.35, w: 7.6, colW: [1.9, 5.7], fontSize: 13, color: C.text1, valign: "middle",
                     border: { type: "solid", pt: 0.75, color: HEX.grid }, rowH: 0.52, margin: 0.06 });
  text(s, "가능한 조합은 수백만 개 이상 → 1단계(가시성)로 거르고 2단계(검출기)로 정밀 평가", { x: 5.1, y: 6.2, w: 7.6, h: 0.45,
       fontSize: 14, bold: true, color: C.accent4 });
  s.addNotes("왼쪽 위는 카메라 후보 자리 96곳을 Isaac Sim에서 그린 것이고, 왼쪽 아래는 회전대입니다. 오른쪽 표가 탐색하는 설정과 센서·공정 모델입니다. 96곳 모두 컨베이어와 부딪히거나 가리지 않음을 Isaac에서 확인했습니다.");
}
{
  const s = newSlide("CONTENT", "검증에 쓴 제품 6종과 결함 6종");
  const notes = { tless_13: "전구 소켓 · 자기 가림 큼", tless_18: "원뿔형 하우징 · 자기 가림 최대", tless_29: "사각 전기 박스 · 단순",
                  tless_30: "원형 전기 박스 · 단순", mulsen_nut: "육각 너트 · 실제 스캔", mulsen_flat_pad: "고리형 패드 · 실제 스캔" };
  D.parts.forEach((p, i) => {
    const x = 0.6 + i * 2.05;
    img(s, path.join(STILL, `thumb_${p}.jpg`), x, 1.3, 1.9, 1.52, D.ko[p]);
    text(s, D.short[p], { x, y: 2.88, w: 1.9, h: 0.3, fontSize: 13, bold: true, align: "center" });
    text(s, notes[p], { x, y: 3.17, w: 1.9, h: 0.3, fontSize: 11, color: C.text2, align: "center" });
  });
  D.types.forEach((t, i) => {
    const x = 0.6 + i * 2.05;
    img(s, path.join(STILL, `defect_${t}.jpg`), x, 3.65, 1.9, 1.36, TYPES_KO[i]);
    text(s, TYPES_KO[i], { x, y: 5.06, w: 1.9, h: 0.3, fontSize: 13, bold: true, align: "center" });
    text(s, D.defect_sizes[t].replace(/ mm, /g, " · "), { x, y: 5.35, w: 1.9, h: 0.3, fontSize: 10, color: C.text2, align: "center" });
  });
  bullets(s, [
    "제품: T-LESS 전기 부품 4종(공개 재배포 가능) + MulSen-AD 실제 스캔 부품 2종(실제 결함 스캔 포함), 모두 가장 긴 변 40 mm. 결함은 T-LESS 29번 예시, 빨강 = 자동 정답 표시",
    "부품당 정상품 140개(학습 40 + 기준선 100) + 결함 72개(탐색용 36 + 성적용 36) + 바닥 결함 대조군 6개. 결함 크기는 기본 배치 미검률 20–60%로 보정 후 고정(실측 52%)",
  ], { x: 0.6, y: 5.85, w: 12.1, h: 1.05, fontSize: 13 });
  s.addNotes("검증 실험은 T-LESS 전기 부품 4종과 MulSen-AD 실제 스캔 부품 2종으로 했습니다. 결함은 부품 형상을 직접 깎거나 밀어서 만들고 정답 표시가 자동으로 붙습니다.");
}
{
  const s = newSlide("CONTENT", "검출기: 정상품만 배우고 처음 보는 결함을 찾는다");
  img(s, path.join(STILL, "capture_tless_29.png"), 0.6, 1.35, 6.4, 3.6, "카메라가 얻는 컬러·깊이 영상");
  text(s, "T-LESS 29번 권장 설정의 카메라 3대가 얻은 컬러·깊이 영상 (회전 전후)", { x: 0.6, y: 5.0, w: 6.4, h: 0.3, fontSize: 12, color: C.text2 });
  bullets(s, [
    { text: "깊이 영상 → 3D 점으로 합치고 설계 모델과 위치를 맞춘다(정합)", options: {} },
    { text: "점마다 주변 형상을 숫자 33개로 요약(FPFH 특징, 1·2·4 mm) + 부품 위 위치", options: {} },
    { text: "메모리 뱅크: 정상품 40개의 특징을 저장 → 새 부품의 각 점이 \"가장 비슷한 정상 특징과 얼마나 다른가\"가 점수", options: {} },
    { text: "판정 기준선 = 정상품 100개 점수의 상위 1%, 신뢰구간은 1,000번 재표집", options: {} },
    { text: "비교 검출기: 설계도 거리(학습 없음), 딥러닝 특징(DINOv2)", options: {} },
  ], { x: 7.3, y: 1.4, w: 5.45, h: 4.6, fontSize: 15 });
  card(s, 0.6, 5.55, 12.15, 1.1);
  text(s, "구현 확인: 공개 실제 데이터(MVTec 3D-AD)에서 공개 검출기를 다시 구현해 dowel 0.883(공개 0.883), cable gland 0.534(공개 0.534/0.551) 재현",
       { x: 0.85, y: 5.7, w: 11.7, h: 0.8, fontSize: 15, bold: true, color: C.text1, valign: "middle" });
  s.addNotes("검출기는 불량 샘플 없이 정상품만으로 만들기 때문에 설정이 바뀌어도 1분 안에 다시 만들 수 있습니다. 직접 구현한 검출기가 공개 논문의 수치를 그대로 재현하는 것도 확인했습니다.");
}
{
  const s = newSlide("CONTENT", "설정 하나를 약 1분에 평가하는 방법: 시점 저장본");
  stat(s, 0.6, 1.3, 3.3, "67분 → 1분", "설정 하나 평가 시간\n(부품 218개 기준)", C.accent2, 32);
  stat(s, 4.1, 1.3, 2.8, "2만 장", "부품당 미리 찍어 둔 깊이 영상\n(4–5분, 약 1.3 GB)", C.accent2, 32);
  stat(s, 7.1, 1.3, 3.0, "0.045 mm", "저장본 조합 vs Isaac 직접 촬영\n점 위치 차 (잡음 0.15 mm보다 작음)", C.accent3, 32);
  stat(s, 10.3, 1.3, 2.5, "0.35 ms", "1단계 가시성 계산\n(설정 하나)", C.accent3, 32);
  bullets(s, [
    "계획서는 부품 60개로 설정당 5–15분을 예상 → 하지만 정상품 20개로는 \"상위 1%\" 기준선을 정할 수 없다(최댓값 하나에 좌우)",
    "부품 수백 개를 설정마다 Isaac으로 찍으면 설정당 약 67분 → 탐색 불가능",
    "해결: 변형 부품을 카메라 자리 96곳에서 한 번씩만 찍어 저장하고, 평가할 때 해당 자리 영상을 골라 센서 잡음·벨트 움직임을 덧입힌다",
    "회전대로 부품을 90° 돌리는 것 = 카메라를 반대로 90° 옮기는 것 → 저장본을 정확히 재사용",
    "대량 촬영은 Isaac과 같은 결과를 내는 빠른 광선 추적(깊이 차 중앙값 0.0001 mm), Isaac은 검증·물리·데모에",
  ], { x: 0.6, y: 3.4, w: 12.1, h: 3.4, fontSize: 15 });
  s.addNotes("이 구조 덕분에 설정 하나를 약 1분에 평가할 수 있게 되었고, 근사가 실제 Isaac 직접 촬영과 같다는 것도 확인했습니다.");
}
{
  const s = newSlide("CONTENT", "실험 원칙: 결과를 믿을 수 있게");
  const P = [
    ["사전 등록", "실험 전에 부품·평가 횟수·판정 기준을 문서로 고정하고 코드 버전을 기록. 결과를 본 뒤 기준을 바꾸지 않는다"],
    ["탐색용·성적용 분리", "설정을 고를 때 쓴 결함으로 성적을 매기지 않는다 (반반 분리)"],
    ["신뢰구간", "결함 1개 = 2.8%p, 신뢰구간 폭 30–40%p를 늘 염두에"],
    ["비교 검출기·대조군", "학습 없는 설계도 거리 검출기로 \"검출기 성질\"과 \"시뮬레이터 오류\"를 가른다"],
    ["이상한 결과 추적", "센서 모델 결함·정합 실패·정렬 버그 모두 \"숫자가 이상하다\"에서 출발"],
  ];
  P.forEach(([h, d], i) => {
    const x = 0.6 + (i % 3) * 4.1, y = 1.4 + Math.floor(i / 3) * 2.65;
    card(s, x, y, 3.85, 2.4);
    numberDot(s, i + 1, x + 0.25, y + 0.25, 0.45);
    text(s, h, { x: x + 0.85, y: y + 0.27, w: 2.8, h: 0.45, fontSize: 17, bold: true });
    text(s, d, { x: x + 0.25, y: y + 0.9, w: 3.35, h: 1.4, fontSize: 14, color: C.text2 });
  });
  s.addNotes("시뮬레이터는 그럴듯한 숫자를 쉽게 만들기 때문에 이 다섯 원칙을 지켰습니다. 실험 도중 바꾼 것은 모두 이유와 함께 사전 등록 문서의 이탈 기록에 남겼습니다.");
}

// ------------------------------------------------------------------ section 3
addSection("무엇을 확인했나");
{
  const s = newSlide("SECTION", "3. 무엇을 확인했나");
  s.addText("점검 9가지 → 개발 중 발견 → 가설 검증 실험 → 견고성 점검", { placeholder: "body" });
}
{
  const s = newSlide("CONTENT", "시뮬레이터·계산 방법 점검 9가지: 모두 통과");
  const hdr = ["점검", "미리 정한 기준", "결과"].map((t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 } } }));
  const rows = [hdr,
    ["Isaac 렌더링 = 빠른 광선 추적 (70장, 452만 화소)", "차이 0.02 mm 이하", "중앙값 0.0001 mm, 화소 겹침 99.993%"],
    ["저장본 조합 = Isaac 직접 촬영 (40개 설정, 1,276장)", "센서 잡음(0.15 mm)보다 작게", "중앙값 0.045 mm"],
    ["결함 정답 표시 면적 (6개 부품)", "오차 10% 이내", "눌림 2.7%, 돌출 1.4%, 구멍 3.6% (규칙 수정 후)"],
    ["설정 하나 평가 시간", "약 1분", "기본 3대 50–70초, 1단계 0.35 ms"],
    ["결함 크기 보정 (6개 부품)", "기본 배치 미검률 20–60%", "평균 52% → 크기 고정"],
    ["검출기 구현 재현 (MVTec 3D-AD)", "공개 수치 ±0.02", "0.883 (공개 0.883), 0.534 (공개 0.534)"],
    ["카메라 후보 자리 설치 가능성", "모두 유효", "96 / 96"],
    ["시제품 기능 유지", "정확도 99% 이상", "100% (원래 50%)"],
    ["최소 동작 점검 (설정 1개 → 성능·비용)", "자동 동작", "통과 (Isaac–저장본 깊이 차 2.5 µm 이하)"],
  ];
  s.addTable(rows, { x: 0.6, y: 1.35, w: 12.1, colW: [4.9, 2.9, 4.3], fontSize: 14, color: C.text1, valign: "middle",
                     border: { type: "solid", pt: 0.75, color: HEX.grid }, rowH: 0.5, margin: 0.07 });
  s.addNotes("계획서에는 시뮬레이터를 믿을 수 있는지 점검하는 단계가 따로 없었습니다. 렌더링, 근사, 정답 표시, 시간, 결함 난이도, 검출기 구현, 설치 가능성, 시제품 기능, 최소 동작까지 아홉 가지를 미리 정한 기준으로 점검해 모두 통과했습니다.");
}
{
  const s = newSlide("CONTENT", "개발 중에 발견하고 바로잡은 것");
  const F = [
    ["센서 모델 결함", "42% → 89%", "카메라 1→2대에서 학습 없는 검출기까지 나빠짐 → 비스듬한 카메라의 \"튀는 점 장막\"과 노출 중 단차 혼합이 원인. 실제 카메라의 경계 필터를 넣어 44/39/36%(1/3/6대)로 정상화"],
    ["위치 맞추기 실패", "미검률 86%", "반복 홈이 있는 부품이 한 칸 밀려 정합 → 정상품이 불량처럼 보여 기준선 상승. 여러 출발점 정합, 회전대는 단계별 정합"],
    ["가시성 편향", "75–97%", "가시성 상위가 전부 \"90°씩 세 번 회전\"이었는데 실제로는 더 나쁨 → 2단계 후보를 카메라 수×회전 방식별로 고르게"],
    ["실제 스캔 정렬 버그", "결함 2 → 29개", "캡슐 같은 회전체에서 정밀 정렬이 10 mm 넘게 발산 → 정렬 방식 수정, 실제 결함 14/23/29개로 재구성"],
  ];
  F.forEach(([h, num, d], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.35 + Math.floor(i / 2) * 2.55;
    card(s, x, y, 5.95, 2.35);
    text(s, h, { x: x + 0.3, y: y + 0.2, w: 3.3, h: 0.45, fontSize: 17, bold: true });
    text(s, num, { x: x + 3.4, y: y + 0.12, w: 2.3, h: 0.55, fontSize: 22, bold: true, color: C.accent4, align: "right" });
    text(s, d, { x: x + 0.3, y: y + 0.8, w: 5.4, h: 1.45, fontSize: 13.5, color: C.text2 });
  });
  text(s, "그 밖에: 결함 정답 표시가 넓던 규칙, 검출기 설정 재선정(개발용 부품으로만), 코어 과다 사용(106초 → 50초), 실험 예산 설정 오류, 요약 표시 오류 2건(판정 영향 없음)",
       { x: 0.6, y: 6.4, w: 12.1, h: 0.5, fontSize: 12.5, color: C.text2 });
  s.addNotes("특히 첫 번째가 중요합니다. 카메라를 늘리면 나빠진다는 결과가 처음에는 흥미로운 발견처럼 보였지만, 학습이 필요 없는 비교 검출기까지 나빠지는 것을 보고 센서 모델의 결함임을 찾아냈습니다. 이 수정 이전 결과는 모두 다시 계산했습니다.");
}
{
  const s = newSlide("CONTENT", "검증 실험 지도");
  const hdr = ["실험", "질문 (가설)", "판정"].map((t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 } } }));
  const V = (t, k) => ({ text: t, options: { bold: true, color: k === "yes" ? C.accent6 : k === "no" ? C.accent4 : C.text2 } });
  const rows = [hdr,
    ["부품 간 최적 설정 교차 적용 실험", "최적 설정은 부품마다 다른가", V("기준 미달", "no")],
    ["가시성 대리지표 타당성 실험", "가시성만으로 설정을 골라도 되는가", V("기준 미달", "no")],
    ["카메라 추가 효과 실험", "카메라를 더하면 오히려 나빠지는가 (검출기 고유 성질)", V("기준 미달", "no")],
    ["정지 촬영 대 인라인 공정 비교 실험", "공정을 반영하면 권장 설정이 달라지는가", V("지지 (3/3)", "yes")],
    ["실제 스캔 결함 순위 유지 실험", "시뮬레이션 순위가 실제 결함에서도 유지되는가", V("재실행 예정", "wip")],
    ["검출기 교체 강건성 실험", "검출기를 바꿔도 같은 설정이 좋은가", V("1/2 부품 강건", "wip")],
    ["실제 센서 비교 실험 (MVTec 3D-AD)", "시뮬레이터 센서는 실제와 얼마나 다른가", V("실제가 훨씬 깨끗", "wip")],
    ["센서 잡음 수준 비교 실험", "잡음 수준을 바꿔도 순위가 유지되는가", V("중간 정도 유지", "wip")],
    ["부품 놓임 방향 실험", "부품이 아무 방향으로 놓이면?", V("미검률 44→89%", "wip")],
    ["탐색 방법 효율 비교 실험", "두 단계 탐색은 효율적인가", V("진행 중 (+59%)", "wip")],
  ];
  s.addTable(rows, { x: 0.6, y: 1.35, w: 12.1, colW: [4.5, 5.5, 2.1], fontSize: 14, color: C.text1, valign: "middle",
                     border: { type: "solid", pt: 0.75, color: HEX.grid }, rowH: 0.47, margin: 0.07 });
  s.addNotes("위 다섯 줄이 가설 검증 실험, 아래 다섯 줄이 결론의 견고성을 보는 실험입니다. 다음 장부터 하나씩 설명합니다.");
}
{
  const s = newSlide("CONTENT", "① 최적 설정은 부품마다 다른가");
  badge(s, "기준 미달", 11.35, 0.5, "no");
  const M = D.cross.M, P = D.parts;
  const color = (d) => {
    const t = Math.max(-1, Math.min(1, d / 0.4));
    const mix = (a, b, u) => a.map((v, i) => Math.round(v + (b[i] - v) * u));
    const mid = [240, 239, 236], blue = [42, 120, 214], red = [227, 73, 72];
    const c = t >= 0 ? mix(mid, red, t) : mix(mid, blue, -t);
    return c.map((v) => v.toString(16).padStart(2, "0")).join("").toUpperCase();
  };
  const head = [{ text: "최적 설정 \\ 적용 부품", options: { bold: true, fontSize: 11, color: C.background1, fill: { color: C.text1 } } }]
    .concat(P.map((p) => ({ text: D.short[p], options: { bold: true, fontSize: 11, color: C.background1, fill: { color: C.text1 }, align: "center" } })));
  const rows = [head];
  P.forEach((pa, i) => {
    const r = [{ text: D.short[pa], options: { bold: true, fontSize: 12 } }];
    P.forEach((pb, j) => {
      const d = M[i][j] - M[j][j];
      r.push({ text: i === j ? pct(M[i][j]) : `${d >= 0 ? "+" : ""}${Math.round(d * 100)}`,
               options: { align: "center", bold: i === j, fontSize: 13, fill: { color: i === j ? "FFFFFF" : color(d) },
                          color: Math.abs(d) > 0.25 ? "FFFFFF" : HEX.ink } });
    });
    rows.push(r);
  });
  s.addTable(rows, { x: 0.6, y: 1.4, w: 7.3, colW: [1.6, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95], valign: "middle",
                     border: { type: "solid", pt: 0.75, color: "FFFFFF" }, rowH: 0.6, margin: 0.04 });
  text(s, "대각선 = 자기 최적 설정의 미검률, 나머지 = 그 대비 차이(%p, 빨강 = 손해, 파랑 = 오히려 나음)", { x: 0.6, y: 5.75, w: 7.3, h: 0.5, fontSize: 12, color: C.text2 });
  stat(s, 8.3, 1.35, 2.2, `${(D.cross.median_delta * 100).toFixed(1)}%p`, "손해의 중앙값\n(기준 10%p 이상)", C.accent4);
  stat(s, 10.6, 1.35, 2.1, `${D.cross.n_sig} / ${D.cross.n_pairs}`, "확실한 차이 쌍\n(기준 20쌍 이상)", C.accent4);
  bullets(s, [
    "6개 부품 각각 40회 탐색, 성적용 결함 36개로 평가",
    "확실한 손해: T-LESS 18번 설정 → 29번 +33%p, 29번 → 13번 +33%p",
    "부품 안에서는 선택이 안정적(탐색용·성적용 순위 0.81–0.94)",
    "그러나 자기 설정이 평균 9.3%p 유리한 경향은 있다 → 다음 장",
    "평판 패드 설정 하나를 모두에 쓰면 평균 48.6% = 부품별 최적 평균 48.6%",
  ], { x: 8.3, y: 3.35, w: 4.45, h: 3.4, fontSize: 13.5 });
  s.addNotes("가설 1은 미리 정한 기준에 못 미쳤습니다. 손해 중앙값이 9.7%p로 기준 10%p에 결함 1개 차이로 못 미쳤고, 통계적으로 확실한 쌍은 30쌍 중 2쌍뿐입니다. 다만 표의 대각선을 부품별로 보면 자기 설정이 대체로 상위권이라, 다음 장에서 다시 읽어 봅니다.");
}
{
  const s = newSlide("CONTENT", "① 다시 보기: 자기 최적 설정이 가장 나은 경향");
  badge(s, "탐색적", 11.35, 0.5, "wip");
  img(s, path.join(FIG, "own_rank.png"), 0.5, 1.25, 7.6, 3.8, "부품마다 6개 최적 설정 중 자기 설정의 순위");
  stat(s, 0.6, 5.1, 2.4, "+9.3%p", "다른 부품 설정보다\n덜 놓침 (p = 0.046)", C.accent2, 34);
  stat(s, 3.1, 5.1, 2.5, "3 / 5 · 2 / 5", "평판 패드 제외 시\n1위 · 2위인 부품 수", C.accent2, 34);
  stat(s, 5.75, 5.1, 2.4, "48.6%", "평판 패드 설정 하나로\n= 부품별 최적 평균", C.accent4, 34);
  bullets(s, [
    { text: "기준이 보수적이었나 — 일부 맞다", options: { bold: true } },
    "\"30쌍 중 20쌍 확실\"은 결함 36개(쌍별 신뢰구간 폭 30–40%p)로는 사실상 달성 불가였다",
    "그래도 결과를 본 뒤 기준은 바꾸지 않는다 → 판정은 그대로, 새 실험으로 다시 묻는다",
    { text: "해석과 반대 가능성", options: { bold: true } },
    "평판 패드(납작, 한 대로 표면 81%가 보임)는 어떤 설정이든 잘 된다 → 빼면 자기 설정 우위가 뚜렷",
    "탐색을 늘리면 우위가 강해질 수도, 모두 범용 설정 하나로 모일 수도 있다 → 실험으로 가른다",
  ], { x: 8.4, y: 1.35, w: 4.4, h: 5.4, fontSize: 13 });
  s.addNotes("같은 결과를 부품별로 다시 읽으면, 자기 최적 설정이 다른 부품 설정의 평균보다 9.3%p 덜 놓치고, 이 차이가 우연일 확률은 약 5%입니다. 평판 패드를 빼면 다섯 개 중 세 개에서 자기 설정이 1위, 나머지 두 개도 결함 1–3개 차이의 2위입니다. 다만 평판 패드의 최적 설정 하나를 모든 부품에 써도 부품별 최적과 평균이 같아서, 범용 설정 하나로 충분할 가능성도 남아 있습니다. 미리 정한 기준은 표본 크기 때문에 지나치게 엄격했지만, 결과를 본 뒤 기준을 바꾸지 않고 새 실험으로 다시 묻겠습니다. 이 장은 탐색적 분석입니다.");
}
{
  const s = newSlide("CONTENT", "어떤 부품이 남의 설정에 손해를 보나");
  badge(s, "탐색적", 11.35, 0.5, "wip");
  img(s, path.join(FIG, "part_features.png"), 0.5, 1.25, 7.4, 4.1, "90° 회전 이득과 손해");
  const hdr = ["부품", "손해", "90° 회전 이득", "최적 설정"].map((t) => ({ text: t, options: { bold: true, fontSize: 11.5, color: C.background1, fill: { color: C.text1 } } }));
  const R = [["T-LESS 29", "+19%p", "+21%p", "3대 · 90° 회전"], ["T-LESS 13", "+18%p", "+24%p", "1대 · 90° 회전"],
             ["너트", "+17%p", "+17%p", "2대 · 90° 회전"], ["T-LESS 18", "+6%p", "+16%p", "3대 · 회전 없음"],
             ["T-LESS 30", "+2%p", "+20%p", "1대 · 90° 회전"], ["평판 패드", "-6%p", "+10%p", "2대 · 회전 없음"]];
  s.addTable([hdr].concat(R), { x: 8.2, y: 1.35, w: 4.55, colW: [1.15, 0.8, 1.2, 1.4], fontSize: 11.5, color: C.text1,
                               valign: "middle", border: { type: "solid", pt: 0.75, color: HEX.grid }, rowH: 0.42, margin: 0.05 });
  text(s, "90° 회전 이득 = 가장 좋은 카메라 1대에 90° 회전을 더하면 더 보이는 표면. 부품 6개 탐색이라 결론이 아니라 가설",
       { x: 8.2, y: 4.45, w: 4.55, h: 0.9, fontSize: 12, color: C.text2 });
  bullets(s, [
    "회전으로 더 보이는 표면이 큰 부품(+17–24%p)은 모두 회전대를 썼다 (순위상관 +0.83)",
    "그런 부품일수록 남의 설정에 손해가 컸다 (+0.77, 30번은 예외). \"한 대로 절반도 못 보는 결함 비율\"은 회전대와는 +0.74, 손해와는 +0.35로 약함",
    "\"특징이 비슷하면 설정을 바꿔 써도 된다\"는 아직 안 보인다 (p = 0.83)",
  ], { x: 0.6, y: 5.5, w: 12.1, h: 1.3, fontSize: 13 });
  s.addNotes("손해가 컸던 부품과 아닌 부품을 나누는 특징을 찾아봤습니다. 가장 좋은 카메라 한 대에 90도 회전을 더했을 때 더 보이는 표면이 큰 부품, 즉 13·29·30번과 너트는 모두 최적 설정에 회전대를 썼고, 그런 부품일수록 다른 부품의 설정을 쓰면 손해가 컸습니다. 30번은 예외입니다. 처음에는 한 대로 잘 안 보이는 결함의 비율로 봤는데, 대조군 결함이 섞여 부품 간 차이가 과장돼 있어 사전 등록한 정의로 다시 계산했고, 그 특징은 손해와의 관계가 약했습니다. 부품이 여섯 개뿐이라 가설로 보고, 다음 장의 실험으로 판정합니다.");
}
{
  const s = newSlide("CONTENT", "이 해석으로 설계한 실험 두 가지");
  badge(s, "사전 등록", 11.35, 0.5, "yes");
  const E = [
    ["부품별 최적 설정 대 범용 설정 비교 실험",
     "충분히 탐색한 자기 최적이 6부품 공통 범용 설정보다 나은가? 탐색을 늘리면 자기 설정 우위가 강해지는가?",
     ["부품별 최적: 교차 적용 확대 재실행(6부품 × 시드 3 × 60회)",
      "범용 설정: 확대 재실행 상위 5개/부품 + 1차 최적 6개(최대 36개) 중 6부품 평균이 가장 좋은 하나",
      "새 성적용 세트: 결함 144개 + 정상품 200개/부품 (선택에 안 씀)",
      "판정: 범용 대비 이득 ≥ 5%p·하한 > 0 → 부품별 특화, 상한 < 5%p → 범용으로 충분",
      "보조 판정: 1차 대비 확대 재실행에서 자기 설정 우위가 커지면 강화",
      "약 300회 평가, 10월 29일 밤 완료 예상"]],
    ["부품 형상 특징과 최적 검사 설정의 관계 실험",
     "어떤 형상 특징이 어떤 최적 설정(회전대, 카메라 수, 고도)을 낳고, 어떤 부품이 남의 설정에 손해를 보는가?",
     ["부품 16개: 지금 6 + 데이터가 있는 4 + T-LESS 16·10·24·8·17·14번(특징 범위로 선정)",
      "특징 7개를 평가 전에 고정: 납작함, 1대 가시, 90° 회전 이득 등",
      "모든 부품에 고정 설정 20개 + 새 부품 탐색 + 16×16 교차 적용",
      "가설 4개: 회전 이득 → 회전 유리, 어려운 결함 → 손해 큼, 1대 가시 → 설정 차이 작음, 비슷한 부품 → 설정 공유",
      "실용 규칙: 특징이 가까운 3개 부품의 최선 설정이 탐색 최적과 5%p 이내인가",
      "약 930회 평가, 10월 31일 완료 예상"]],
  ];
  E.forEach(([name, q, items], i) => {
    const x = 0.6 + i * 6.15;
    card(s, x, 1.3, 5.95, 5.15);
    numberDot(s, i + 1, x + 0.25, 1.5, 0.45);
    text(s, name, { x: x + 0.85, y: 1.48, w: 4.95, h: 0.55, fontSize: 16, bold: true, valign: "middle" });
    text(s, q, { x: x + 0.3, y: 2.15, w: 5.4, h: 0.85, fontSize: 13, color: C.accent2, bold: true });
    bullets(s, items, { x: x + 0.3, y: 3.05, w: 5.4, h: 3.3, fontSize: 12 });
  });
  text(s, "1차 교차 적용 실험의 판정(기준 미달)은 그대로 두고, 두 실험은 실행 전에 판정 규칙까지 고정해 대기열에 넣었다. 날짜는 추정(앞 실험은 1.5–2배 걸린 적 있음).",
       { x: 0.6, y: 6.55, w: 12.1, h: 0.4, fontSize: 12, color: C.text2 });
  s.addNotes("이 해석을 확인하려고 두 실험을 설계했고, 실험 대기열을 관리하는 쪽에서 판정 규칙까지 미리 고정해 대기열에 넣었습니다. 첫째는 충분히 탐색한 부품별 최적 설정과 여섯 부품 공통의 범용 설정을 직접 비교하는 실험입니다. 결함 144개와 정상품 200개로 검정력을 확보했고, 탐색을 늘렸을 때 자기 설정의 우위가 강해지는지도 따로 판정합니다. 둘째는 부품을 16개로 늘려, 형상 특징이 어떤 최적 설정을 낳는지 보는 실험입니다. 결과는 이런 특징의 부품에는 이런 설정이라는 지도로 정리해 기말 결론으로 삼습니다. 두 실험 모두 10월 말 완료 예정입니다.");
}
{
  const s = newSlide("CONTENT", "② 가시성만으로 설정을 골라도 되는가");
  badge(s, "기준 미달", 11.35, 0.5, "no");
  ["tless_13", "tless_18", "tless_29", "tless_30"].forEach((p, k) => {
    const v = D.visibility[p];
    const xs = v.pts.map((q) => q[0]), ys = v.pts.map((q) => q[1]);
    s.addChart(pres.charts.SCATTER, [{ name: "가시성", values: xs }, { name: "미검률", values: ys }],
      Object.assign({}, AX, { x: 0.5 + k * 3.1, y: 1.3, w: 3.0, h: 2.9, chartColors: [k === 3 ? HEX.series[2] : HEX.series[0]],
        lineSize: 0, lineDataSymbol: "circle", lineDataSymbolSize: 5, showLegend: false, showTitle: true,
        title: `${D.short[p]} 순위상관 ${v.rho.toFixed(2)}`, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0%",
        catAxisMinVal: 0, catAxisMaxVal: 0.8, catAxisLabelFormatCode: "0.0", valAxisMajorUnit: 0.25, catAxisMajorUnit: 0.2,
        showValAxisTitle: k === 0, valAxisTitle: "미검률", valAxisTitleFontSize: 11, valAxisTitleColor: HEX.ink2,
        showCatAxisTitle: true, catAxisTitle: "가시성", catAxisTitleFontSize: 11, catAxisTitleColor: HEX.ink2,
        objectName: `visibility scatter ${p}` }));
  });
  text(s, "기준: 순위상관 -0.5 이하 + 실제 상위 10개 중 70%가 가시성 상위 30개 안, 4개 중 3개 부품 → 충족은 T-LESS 30번 하나뿐 (초록)",
       { x: 0.6, y: 4.3, w: 12.1, h: 0.4, fontSize: 13, color: C.text2 });
  card(s, 0.6, 4.85, 5.95, 1.85);
  text(s, "보이는데 놓친다", { x: 0.85, y: 4.98, w: 5.5, h: 0.4, fontSize: 17, bold: true });
  text(s, "가시성 상위 1/3 설정에서 결함 36개 중 32–34개가 이미 보이는데 57–85%를 놓쳤다. 구멍은 5–36%로 잘 잡지만 깊이 0.2–1.0 mm의 눌림·돌출·균열은 65–99%를 놓친다.",
       { x: 0.85, y: 5.45, w: 5.5, h: 1.2, fontSize: 13.5, color: C.text2 });
  card(s, 6.8, 4.85, 5.95, 1.85);
  text(s, "더 중요한 것: 회전대 사용과 위치 맞추기 품질", { x: 7.05, y: 4.98, w: 5.5, h: 0.4, fontSize: 17, bold: true });
  text(s, "미검률과의 순위상관이 회전대 사용 최대 -0.65, 정합 품질 -0.37 ~ -0.58. 카메라 수 자체는 거의 무관. 잡음을 낮춰도 가시성의 설명력은 커지지 않았다.",
       { x: 7.05, y: 5.45, w: 5.5, h: 1.2, fontSize: 13.5, color: C.text2 });
  s.addNotes("가시성 대리지표 가설은 네 부품 중 하나에서만 기준을 충족했습니다. 결함이 보이는 것과 잡히는 것은 다릅니다. 그래서 1단계 가시성 거르기만으로는 부족하고, 검출기로 직접 평가하는 2단계가 꼭 필요합니다.");
}
{
  const s = newSlide("CONTENT", "③ 카메라를 더하면 오히려 나빠지는가");
  badge(s, "기준 미달", 11.35, 0.5, "no");
  const names = { greedy_vis: "가시성 기준 한 대씩 추가", greedy_fnr: "미검률 기준 한 대씩 추가", random: "무작위", bo_warm: "베이지안 최적화" };
  ["tless_13", "mulsen_nut"].forEach((p, k) => {
    const cur = D.camera_addition[p].curves;
    const labels = ["1", "2", "3", "4", "5", "6"];
    const series = Object.keys(names).map((m) => ({ name: names[m], labels,
      values: labels.map((n) => (cur[m][n] === null || cur[m][n] === undefined ? null : cur[m][n])) }));
    s.addChart(pres.charts.LINE, series, Object.assign({}, AX, { x: 0.5 + k * 4.0, y: 1.3, w: 3.9, h: 3.6,
      chartColors: HEX.series.slice(0, 4), lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 6,
      showLegend: k === 1, legendPos: "b", showTitle: true, title: `${D.short[p]} (3대에서 최적화 이득 ${Math.round(D.camera_addition[p].gain * 100)}%p)`,
      valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0%", valAxisMajorUnit: 0.25,
      showCatAxisTitle: true, catAxisTitle: "카메라 수", catAxisTitleFontSize: 11, catAxisTitleColor: HEX.ink2,
      objectName: `camera addition ${p}` }));
  });
  const hdr = ["조건", "결과", "기준"].map((t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 } } }));
  s.addTable([hdr,
    ["최적화가 가시성 기준보다 5%p 이상 나은 부품", { text: "4 / 6 ✓", options: { bold: true, color: C.accent6 } }, "4 / 6 이상"],
    ["카메라 1대 추가 시 나빠지는 비율 (메모리 뱅크)", { text: "42% ✓", options: { bold: true, color: C.accent6 } }, "10% 초과"],
    ["같은 비율 (설계도 거리 = 대조군)", { text: "31% ✗", options: { bold: true, color: C.accent4 } }, "3% 미만"],
  ], { x: 8.65, y: 1.35, w: 4.1, colW: [2.2, 0.95, 0.95], fontSize: 12, color: C.text1, valign: "middle",
       border: { type: "solid", pt: 0.75, color: HEX.grid }, rowH: 0.7, margin: 0.05 });
  bullets(s, [
    "대조군도 31% 나빠짐 → 메모리 뱅크 고유 성질이라는 근거 없음",
    "카메라 추가 효과는 좋아짐 ≈ 나빠짐, 평균 0 (2%p 기준은 결함 1개 2.8%p보다 작아 사실상 달성 불가였음)",
    "차이를 만든 것은 탐색 방법이 아니라 목적: 미검률 기준 탐욕적 추가도 최적화와 비슷 → \"검출 성능으로 직접 찾아야 한다\"",
  ], { x: 0.6, y: 5.05, w: 12.1, h: 1.8, fontSize: 14 });
  s.addNotes("카메라 추가 효과 실험의 세 조건 중 둘은 충족했지만, 비교용 설계도 거리 검출기도 카메라를 더하면 31% 나빠져 메모리 뱅크 고유의 성질이라는 근거가 성립하지 않았습니다. 다만 가시성 기준보다 검출 성능 기준 탐색이 낫다는 것은 확인했습니다.");
}
{
  const s = newSlide("CONTENT", "④ 공정을 반영하면 권장 설정이 달라지는가");
  badge(s, "지지 3/3", 11.35, 0.5, "yes");
  const parts = ["tless_13", "tless_29", "tless_30"];
  const labels = parts.map((p) => `${D.short[p]}`);
  s.addChart(pres.charts.BAR, [
    { name: "정지 촬영 최적을 라인에 그대로 (사이클 초과)", labels, values: parts.map((p) => D.static_inline[p].static_fnr) },
    { name: "인라인 최적 (사이클 1.4초)", labels, values: parts.map((p) => D.static_inline[p].inline_fnr) },
  ], Object.assign({}, AX, { x: 0.5, y: 1.3, w: 6.6, h: 4.3, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [HEX.series[1], HEX.series[0]], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0%",
    showLegend: true, legendPos: "b", showTitle: true, title: "미검률 (0.6 m/s)", valAxisMinVal: 0, valAxisMaxVal: 0.8,
    valAxisLabelFormatCode: "0%", objectName: "static vs inline" }));
  const hdr = ["제품", "정지 촬영 최적", "인라인 최적", "정지 최적의 사이클"].map((t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 } } }));
  s.addTable([hdr].concat(parts.map((p) => {
    const v = D.static_inline[p];
    return [D.short[p], `${v.static_n}대, ${v.static_rot}`, `${v.inline_n}대, ${v.inline_rot}`,
            { text: `${v.static_T.toFixed(1)}초 > 1.5초`, options: { bold: true, color: C.accent4 } }];
  })), { x: 7.3, y: 1.35, w: 5.45, colW: [1.05, 1.4, 1.35, 1.65], fontSize: 12, color: C.text1, valign: "middle",
         border: { type: "solid", pt: 0.75, color: HEX.grid }, rowH: 0.55, margin: 0.05 });
  bullets(s, [
    "3개 제품 모두 권장 회전 방식이 달라지고, 정지 최적은 가장 빠른 벨트에서도 1.5초 제한을 넘는다",
    "예상과 달리 미검률은 정지 최적이 8–42%p 더 낮다 → 설정이 바뀌는 이유는 움직임이 아니라 사이클 시간 제한",
    "번짐(1.2 mm)·촬영 시점 오차(3 mm)를 하나씩 빼도 결함 1–3개(±8%p) 차이, 방향도 일정하지 않음",
  ], { x: 7.4, y: 3.75, w: 5.35, h: 3.0, fontSize: 13.5 });
  s.addNotes("가설 4는 세 제품 모두에서 지지되었습니다. 그런데 이유가 예상과 달랐습니다. 움직임이 검출을 망쳐서가 아니라, 회전대 시간 때문에 정지 촬영 최적이 사이클 1.5초를 넘어서입니다. 그래서 주장은 '사이클 시간 제한을 탐색에 넣어야 한다'로 정리합니다.");
}
{
  const s = newSlide("CONTENT", "⑤ 시뮬레이션 순위가 실제 결함에서도 유지되는가");
  badge(s, "재실행 예정", 11.35, 0.5, "wip");
  const steps = [
    ["설계", "MulSen-AD 실제 결함 스캔을 기준 모델에 정렬해 같은 가상 공정에 넣고, 합성 결함으로 고른 설정 40개의 순위를 비교 (기준: 순위 일치도 0.5 이상 + 상위 5개 중 3개 겹침)"],
    ["첫 실행 중단 (10/23)", "6시간 뒤 캡슐에서 오류. 근본 원인: 실제 스캔 정렬 대부분 실패 → 실제 결함이 너트 11 · 평판 패드 13 · 캡슐 2개만 남음"],
    ["원인", "캡슐 같은 회전체에서 정밀 정렬(점–평면 방식)이 완벽한 시작 위치에서도 10 mm 넘게 발산. 평판 패드는 나쁜 국소해에 멈춤"],
    ["수정", "주축 정렬 시작값 추가, 점–점 방식 먼저, 점–평면은 품질이 떨어지지 않을 때만, 결함 부위를 빼고 정렬, 수용 기준 \"4복셀 안 95%\""],
  ];
  steps.forEach(([h, d], i) => {
    const y = 1.35 + i * 1.12;
    numberDot(s, i + 1, 0.6, y + 0.08, 0.45);
    text(s, h, { x: 1.25, y: y + 0.05, w: 2.6, h: 0.5, fontSize: 16, bold: true });
    text(s, d, { x: 3.9, y: y, w: 8.85, h: 1.0, fontSize: 13.5, color: C.text2 });
  });
  stat(s, 0.9, 5.7, 3.6, "11 → 14개", "너트 실제 결함 (GT 있는 것 전부)", C.accent2);
  stat(s, 4.9, 5.7, 3.6, "13 → 23개", "평판 패드 실제 결함 (전부)", C.accent2);
  stat(s, 8.9, 5.7, 3.6, "2 → 29개", "캡슐 실제 결함 (33개 중)", C.accent2);
  s.addNotes("가설 3은 아직 답이 없습니다. 첫 실행이 실제 스캔 정렬 버그로 중단되었고, 원인을 찾아 고쳐 실제 결함 수를 크게 늘렸습니다. 수정이 결과에 끌려가지 않도록 중단된 실행의 부분 결과는 열어 보지 않았고, 현재 실험이 끝나면 자동으로 다시 실행됩니다.");
}
{
  const s = newSlide("CONTENT", "결론은 얼마나 튼튼한가: 검출기 · 잡음 · 실제 센서");
  // detectors
  text(s, "검출기를 바꾸면", { x: 0.6, y: 1.3, w: 3.9, h: 0.45, fontSize: 18, bold: true, color: C.accent2 });
  ["tless_13", "tless_29"].forEach((p, k) => {
    const v = D.detectors[p];
    card(s, 0.6, 1.9 + k * 1.75, 3.9, 1.55);
    text(s, D.short[p], { x: 0.85, y: 2.0 + k * 1.75, w: 1.7, h: 0.45, fontSize: 16, bold: true });
    badge(s, v.robust ? "강건" : "달라짐", 2.95, 2.02 + k * 1.75, v.robust ? "yes" : "no");
    text(s, `메모리 뱅크와 설계도 거리의 순위 일치도 ${v.tau.toFixed(2)} (기준 0.5)`, { x: 0.85, y: 2.6 + k * 1.75, w: 3.5, h: 0.75, fontSize: 13, color: C.text2 });
  });
  text(s, "→ 권장 설정에 검출기를 함께 적는다. 학습 없는 설계도 거리 검출기가 평균 13–26%p 덜 놓침(시뮬레이터의 완벽한 설계 모델 덕분)",
       { x: 0.6, y: 5.45, w: 3.9, h: 1.35, fontSize: 12.5, color: C.text2 });
  // noise
  const lv = ["N1", "N2", "N3"], lvKo = ["약", "기본", "강"];
  s.addChart(pres.charts.LINE, ["tless_13", "tless_29"].map((p) => ({ name: D.short[p], labels: lvKo,
      values: lv.map((l) => D.noise[p][l].mean) })),
    Object.assign({}, AX, { x: 4.75, y: 1.3, w: 4.0, h: 3.9, chartColors: HEX.series.slice(0, 2), lineSize: 2,
      lineDataSymbol: "circle", lineDataSymbolSize: 7, showLegend: true, legendPos: "b", showTitle: true,
      title: "잡음 수준별 평균 미검률", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0%", valAxisMajorUnit: 0.25,
      objectName: "noise levels" }));
  text(s, "잡음이 커질수록 단조롭게 증가. 기본 대비 순위 일치도 0.48–0.80, 기본 최적은 다른 수준에서도 대부분 상위 10개 안. 노출 10배 다중 촬영은 거의 영향 없음",
       { x: 4.75, y: 5.35, w: 4.0, h: 1.45, fontSize: 12.5, color: C.text2 });
  // sensor
  const condKo = ["실제 스캔", "잡음 없음", "약(실제 거리)", "기본(실제 거리)", "강(실제 거리)", "기본(0.55 m로 늘림)"];
  s.addChart(pres.charts.BAR, [{ name: "공개 기준 검출기", labels: condKo, values: D.sensor.btf }],
    Object.assign({}, AX, { x: 9.0, y: 1.3, w: 3.8, h: 3.9, barDir: "bar", chartColors: [HEX.series[0]], showLegend: false,
      showTitle: true, title: "실제 스캔 다시 찍기: 분리 능력", valAxisMinVal: 0.4, valAxisMaxVal: 1, valAxisMajorUnit: 0.2,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", catAxisOrientation: "maxMin",
      catAxisLabelFontSize: 10, objectName: "sensor gap" }));
  text(s, "실제 센서(부품 위 잡음 0.003 mm)가 우리 기본(0.14–0.29 mm)보다 훨씬 깨끗 → 우리 센서는 보수적 가정. 절대 수치가 아니라 설정 간 순위를 주장",
       { x: 9.0, y: 5.35, w: 3.8, h: 1.45, fontSize: 12.5, color: C.text2 });
  s.addNotes("결론의 견고성입니다. 검출기를 바꾸면 두 부품 중 하나에서만 순위가 유지되어, 권장 설정에는 검출기를 명시해야 합니다. 잡음 수준을 바꾸면 순위가 중간 정도 유지됩니다. 실제 3D 카메라는 우리 기본 센서보다 훨씬 깨끗해서, 우리 수치는 보수적인 값이고 주장은 순위로 한정합니다.");
}
{
  const s = newSlide("CONTENT", "부품 놓임 방향과 탐색 효율");
  text(s, "부품이 아무 방향으로 놓이면 (T-LESS 29번)", { x: 0.6, y: 1.3, w: 6.0, h: 0.45, fontSize: 18, bold: true, color: C.accent2 });
  const hdr = ["공급 방식", "권장 설정", "미검률", "정합 품질"].map((t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 } } }));
  s.addTable([hdr,
    ["방향 맞춤 (±3°)", "3대 + 90° 회전, 0.6 m/s", { text: "44%", options: { bold: true } }, "0.975"],
    ["방향 무작위", "1대, 회전 없음, 0.6 m/s", { text: "89%", options: { bold: true, color: C.accent4 } }, "0.851"],
  ], { x: 0.6, y: 1.9, w: 6.0, colW: [1.55, 2.45, 0.9, 1.1], fontSize: 13, color: C.text1, valign: "middle",
       border: { type: "solid", pt: 0.75, color: HEX.grid }, rowH: 0.6, margin: 0.05 });
  bullets(s, [
    "방향이 무작위면 검출이 거의 무너진다 (분리 능력 0.56)",
    "원인: 측정 점을 설계 모델에 맞추는 단계가 \"방향 맞춤 공급\"을 전제로 작은 회전(±8°)만 시도",
    "ISCO 결과는 방향 맞춤 공급(진동 피더·가이드 레일) 또는 전역 정합을 전제로 한다 → 한계로 명시",
  ], { x: 0.6, y: 3.95, w: 6.0, h: 2.8, fontSize: 14 });
  const cur = D.search.curves, names = { bo_warm: "1단계 후보로 시작한 최적화", bo_cold: "처음부터 최적화", nsga2: "유전 알고리즘", random: "무작위" };
  const labels = cur.bo_warm.map((_, i) => ((i + 1) % 10 === 0 || i === 0 ? String(i + 1) : " "));
  s.addChart(pres.charts.LINE, Object.keys(names).map((m) => ({ name: names[m], labels, values: cur[m] })),
    Object.assign({}, AX, { x: 6.9, y: 1.3, w: 5.9, h: 4.3, chartColors: HEX.series.slice(0, 4), lineSize: 2,
      lineDataSymbol: "none", showLegend: true, legendPos: "b", showTitle: true,
      title: "탐색 효율 (T-LESS 13번, 진행 중): 찾은 성능–비용 곡선의 넓이", valAxisMinVal: 0, valAxisMaxVal: 0.8,
      valAxisMajorUnit: 0.2, catAxisLabelFrequency: 5, showCatAxisTitle: true, catAxisTitle: "검출기로 평가한 설정 수",
      catAxisTitleFontSize: 11, catAxisTitleColor: HEX.ink2, objectName: "search efficiency" }));
  text(s, `평가 30회 시점: 1단계로 시작한 최적화 ${D.search.at30.bo_warm.toFixed(2)} vs 무작위 ${D.search.at30.random.toFixed(2)} (+${Math.round((D.search.at30.bo_warm / D.search.at30.random - 1) * 100)}%, 기준 +10%). 가시성이 미검률을 잘 예측하지 못해도, 1단계가 만든 다양한 후보는 좋은 출발점이 된다`,
       { x: 6.9, y: 5.7, w: 5.9, h: 1.1, fontSize: 12.5, color: C.text2 });
  s.addNotes("놓임 방향 실험은 우리 정합 단계가 방향 맞춤 공급을 전제로 한다는 한계를 보여 줍니다. 탐색 효율 실험은 아직 진행 중인데, 한 부품의 중간 결과로는 1단계 후보로 시작한 최적화가 무작위보다 30회 시점에서 59% 앞섭니다.");
}

// ------------------------------------------------------------------ section 4
addSection("결과: 제품별 권장 설정");
{
  const s = newSlide("SECTION", "4. 결과: 제품별 권장 검사 설정");
  s.addText("제약: 카메라 3대 이하 · 사이클 1.5초 이하 · 기본 잡음 · 인라인 공정", { placeholder: "body" });
}
{
  const s = newSlide("CONTENT", "제품별 권장 검사 설정");
  const hdr = ["제품", "카메라 (방위/고도/거리)", "회전대", "벨트", "사이클", "미검률 (95% 구간)", "분리 능력"]
    .map((t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 } } }));
  const rows = [hdr].concat(D.recommended.map((r) => [
    { text: D.ko[r.part], options: { bold: true } },
    `${r.n}대: ` + r.cams.map((c) => `${c[0]}°/${c[1]}°/${c[2].toFixed(2)} m`).join(", "),
    r.rotation, `${r.speed} m/s`, `${r.cycle.toFixed(1)}초`,
    { text: `${pct(r.fnr)} (${pct(r.ci[0])}–${pct(r.ci[1])})`, options: { bold: true } }, r.auroc.toFixed(2),
  ]));
  s.addTable(rows, { x: 0.6, y: 1.35, w: 12.1, colW: [2.1, 4.1, 1.2, 1.0, 0.9, 1.8, 1.0], fontSize: 12.5, color: C.text1,
                     valign: "middle", border: { type: "solid", pt: 0.75, color: HEX.grid }, rowH: 0.56, margin: 0.06 });
  bullets(s, [
    "6개 중 4개가 \"0.6 m/s + 90° 한 번 회전\" — 같은 사이클이면 카메라를 늘리기보다 회전대로 시점을 늘리는 선택이 자주 이겼다",
    "계획서의 예상(단순하면 적게, 복잡하면 많이)과 꼭 맞지 않음: 자기 가림 큰 13번은 1대+회전, 단순한 29번은 3대+회전",
    "절대 미검률(31–67%)은 보수적 센서와 어렵게 보정한 결함 아래의 값 → 의미 있는 것은 같은 조건에서의 설정 간 비교",
  ], { x: 0.6, y: 5.45, w: 12.1, h: 1.45, fontSize: 13.5 });
  s.addNotes("방위 0도는 벨트 진행 방향, 고도 90도는 바로 위입니다. 6개 중 4개 제품이 0.6 m/s와 90도 회전 한 번을 선택했습니다. 미검률의 신뢰구간이 넓은 것은 성적용 결함이 36개이기 때문입니다.");
}
{
  const s = newSlide("CONTENT", "권장 설정의 카메라 위치 (위에서 본 그림)");
  img(s, path.join(FIG, "recommended_rigs.png"), 1.2, 1.2, 9.4, 5.75, "부품별 권장 카메라 위치");
  text(s, "중심 = 바로 위(고도 90°), 바깥 원 = 낮은 각도(고도 20°), 큰 표시 = 거리 0.25 m, 작은 표시 = 0.40 m, 회색 점 = 후보 자리 96곳, 화살표 = 벨트 진행 방향(방위 0°)",
       { x: 10.75, y: 1.4, w: 2.0, h: 3.6, fontSize: 12.5, color: C.text2 });
  s.addNotes("제품마다 고른 카메라 위치를 위에서 본 그림입니다. 바로 위에서 보는 카메라 하나에 회전대를 더한 설정, 낮은 각도의 카메라를 마주 보게 둔 설정 등 제품마다 다른 답이 나왔지만, 다른 제품의 설정을 써도 평균적으로는 크게 나빠지지 않았다는 것이 교차 적용 실험의 결과였습니다.");
}
{
  const s = newSlide("CONTENT", "제품별 검출 예시: 권장 설정으로 찍고, 정상품만 배운 검출기로 찾는다");
  img(s, path.join(FIG, "detection_examples.png"), 0.9, 1.2, 9.2, 5.8, "제품별 검출 예시");
  text(s, "각 제품의 왼쪽 = 시뮬레이터가 넣은 결함(빨강, 정답), 오른쪽 = 검출기가 점마다 계산한 이상 점수(노랑·빨강일수록 이상). 6개 제품 모두 예시 결함의 점수가 판정 기준선(정상품 오판정 1%)을 크게 넘어 검출되었다.",
       { x: 10.3, y: 1.4, w: 2.45, h: 4.2, fontSize: 12.5, color: C.text2 });
  s.addNotes("실제 평가와 같은 검출 파이프라인으로 계산한 점 단위 점수입니다. 결함 위치에서 점수가 높게 나타납니다. 다만 결함 종류에 따라 놓치는 경우도 많아서, 다음 장에서 결함 종류별 미검률을 보여 드립니다.");
}
{
  const s = newSlide("CONTENT", "결함 종류별 미검률: 재료가 빠진 결함은 잘, 미세 변형은 어렵게");
  const labels = TYPES_KO;
  s.addChart(pres.charts.BAR, D.recommended.map((r) => ({ name: D.short[r.part], labels, values: r.by_type })),
    Object.assign({}, AX, { x: 0.5, y: 1.3, w: 8.4, h: 5.4, barDir: "col", barGrouping: "clustered", barGapWidthPct: 50,
      chartColors: HEX.series, showLegend: true, legendPos: "b", showTitle: true, title: "권장 설정에서 결함 종류별로 놓친 비율 (결함 6개씩)",
      valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0%", valAxisMajorUnit: 0.25, objectName: "miss by type" }));
  const mean = [0, 1, 2, 3, 4, 5].map((i) => D.recommended.reduce((a, r) => a + r.by_type[i], 0) / D.recommended.length);
  const order = [2, 3, 1, 5, 0, 4];
  order.forEach((i, k) => {
    const y = 1.45 + k * 0.82;
    card(s, 9.2, y, 3.55, 0.7);
    text(s, labels[i], { x: 9.45, y: y + 0.1, w: 1.5, h: 0.5, fontSize: 16, bold: true, valign: "middle" });
    text(s, `평균 ${pct(mean[i])}`, { x: 10.9, y: y + 0.1, w: 1.7, h: 0.5, fontSize: 16, bold: true, align: "right", valign: "middle",
                                    color: mean[i] < 0.34 ? C.accent6 : mean[i] < 0.6 ? C.accent1 : C.accent4 });
  });
  text(s, "기본 잡음(0.15 mm) 아래에서 깊이 0.2–1.0 mm의 미세 변형은 놓치기 쉽다. 휨은 점 주변 형상 특징과 전체 위치 맞추기가 흡수해 버리는 구조적 한계",
       { x: 9.2, y: 6.4, w: 3.55, h: 0.6, fontSize: 11, color: C.text2 });
  s.addNotes("구멍과 절단처럼 재료가 빠진 결함은 잘 잡고, 균열, 눌림, 휨 같은 미세 변형은 어렵습니다. 결과를 결함 종류별로 그대로 밝힐 계획입니다.");
}
{
  const s = newSlide("CONTENT", "데모 영상: 제품별 검출 시뮬레이션과 공정 환경 설정");
  const chapters = ["가상 공정 환경 (Isaac Sim 컨베이어)", "검사할 제품 6종", "결함 생성기 6종", "공정 설정: 카메라 자리 96곳 · 회전대 · 속도",
                    "설정 찾기: 1단계 → 2단계", "제품별 검사 시뮬레이션 6종 (촬영 → 검출 → 성적)", "검증 실험 결과 요약"];
  if (fs.existsSync(VIDEO)) {
    s.addMedia({ type: "video", path: VIDEO, x: 0.6, y: 1.3, w: 7.8, h: 4.39, cover: "data:image/png;base64," + fs.readFileSync(path.join(STILL, "video_cover.png")).toString("base64") });
  } else {
    img(s, path.join(STILL, "part_tless_29.jpg"), 0.6, 1.3, 7.8, 4.39, "데모 영상 장면");
  }
  text(s, "영상 순서", { x: 8.75, y: 1.3, w: 4.0, h: 0.45, fontSize: 18, bold: true, color: C.accent2 });
  chapters.forEach((t, i) => {
    numberDot(s, i + 1, 8.75, 1.9 + i * 0.66, 0.42);
    text(s, t, { x: 9.3, y: 1.92 + i * 0.66, w: 3.45, h: 0.5, fontSize: 13.5, valign: "middle" });
  });
  text(s, "파일: docs/reports/ISCO_중간발표_데모영상.mp4 (약 3분 50초, 한국어 자막)", { x: 0.6, y: 5.85, w: 7.8, h: 0.4, fontSize: 13, color: C.text2 });
  s.addNotes("데모 영상은 가상 공정 환경, 제품과 결함, 공정 설정, 탐색 과정, 제품별 검사 시뮬레이션, 검증 결과 순서로 진행됩니다. 제품마다 권장 설정으로 부품이 검사 지점을 지나고, 회전대가 돌고, 카메라가 찍은 영상에서 검출기가 결함을 찾는 과정을 보여 줍니다.");
}

// ------------------------------------------------------------------ section 5
addSection("무엇이 달라졌나");
{
  const s = newSlide("SECTION", "5. 그래서 무엇이 달라졌나");
  s.addText("목표는 그대로, 방법과 주장은 결과에 맞춰 바꿨다", { placeholder: "body" });
}
{
  const s = newSlide("CONTENT", "처음 가설 → 지금의 답");
  const hdr = ["처음 가설", "판정", "지금의 답"].map((t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 } } }));
  const V = (t, k) => ({ text: t, options: { bold: true, align: "center", color: C.background1,
                                              fill: { color: k === "yes" ? C.accent6 : k === "no" ? C.accent4 : C.text2 } } });
  s.addTable([hdr,
    ["최적 설정은 부품마다 다르다", V("기준 미달", "no"), "기준 미달이나 자기 설정이 평균 9%p 유리(탐색적) → 사전 등록한 범용 설정 비교·부품 특징 실험으로 재판정"],
    ["가시성은 검출 성능을 대신할 수 있다", V("기준 미달", "no"), "보여도 놓친다. 잡음 대비 결함 크기·정합 품질이 더 중요 → 2단계 필수, 1단계는 출발점"],
    ["카메라를 더하면 오히려 나빠진다 (검출기 고유)", V("기준 미달", "no"), "대조군도 나빠져 근거 없음. 단 검출 성능 기준 탐색이 가시성 기준보다 낫다"],
    ["공정을 반영하면 권장 설정이 달라진다", V("지지", "yes"), "3/3. 이유는 사이클 시간 제한(지키는 대가 8–42%p), 움직임 영향은 작다"],
    ["시뮬레이션 순위가 실제 결함에서도 유지된다", V("진행 중", "wip"), "정렬 버그 수정 → 실제 결함 14/23/29개로 재실행 예정"],
  ], { x: 0.6, y: 1.3, w: 12.1, colW: [4.3, 1.4, 6.4], fontSize: 13, color: C.text1, valign: "middle",
       border: { type: "solid", pt: 0.75, color: HEX.grid }, rowH: 0.72, margin: 0.06 });
  card(s, 0.6, 6.15, 12.15, 0.7, { fill: C.text1, line: C.text1 });
  text(s, "기각도 결과다: 미리 정한 기준으로 판정했기 때문에, 가설이 틀린 부분이 그대로 새 결론이 된다", { x: 0.85, y: 6.2, w: 11.7, h: 0.6,
       fontSize: 15, bold: true, color: C.background1, valign: "middle" });
  s.addNotes("다섯 가설의 현재 상태입니다. 기준을 미리 고정했기 때문에 기각된 결과도 그대로 의미 있는 결론이 됩니다.");
}
{
  const s = newSlide("CONTENT", "주장이 이렇게 바뀌었다");
  const pairs = [
    ["부품마다 최적 설정이 다르므로 부품마다 시뮬레이션해야 한다", "자기 설정이 나은 경향, 회전으로 더 보이는 표면이 큰 부품일수록 회전대·손해 큼(가설) → 사전 등록한 새 실험으로 판정"],
    ["가시성으로 빠르게 거르면 된다", "가시성은 필요조건에 가깝지만 충분하지 않다. 검출기로 직접 평가하는 2단계가 꼭 필요"],
    ["검출기는 비가산적이라 베이지안 최적화가 필요하다", "검출기의 실제 미검률을 직접 목적으로 탐색해야 한다 (기제는 주장하지 않음)"],
    ["공정 움직임이 검출을 망친다", "사이클 시간 제한을 무시한 계획은 라인에서 못 쓰는 설정을 추천한다"],
    ["시뮬레이션 미검률이 실제와 비슷하다", "센서는 보수적 가정, 절대 수치가 아니라 설정 간 순위를 주장. 검출기를 함께 명시"],
  ];
  text(s, "처음 생각", { x: 0.6, y: 1.3, w: 5.4, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  text(s, "지금의 주장", { x: 6.85, y: 1.3, w: 5.9, h: 0.4, fontSize: 16, bold: true, color: C.accent2 });
  pairs.forEach(([a, b], i) => {
    const y = 1.8 + i * 1.0;
    card(s, 0.6, y, 5.6, 0.85);
    text(s, a, { x: 0.8, y: y + 0.08, w: 5.2, h: 0.7, fontSize: 13.5, color: C.text2, valign: "middle" });
    s.addShape(pres.shapes.RIGHT_ARROW, { x: 6.3, y: y + 0.27, w: 0.4, h: 0.32, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 } });
    card(s, 6.85, y, 5.9, 0.85, { fill: C.background1, line: C.accent2 });
    text(s, b, { x: 7.05, y: y + 0.08, w: 5.5, h: 0.7, fontSize: 13.5, bold: true, valign: "middle" });
  });
  s.addNotes("결과 때문에 바뀐 주장입니다. 지지되는 핵심은 두 가지, 검출기 성능으로 직접 탐색해야 한다는 것과 사이클 시간 제한을 넣어야 한다는 것입니다.");
}
{
  const s = newSlide("CONTENT", "방법에서 달라진 점 (목표는 그대로)");
  const hdr = ["항목", "계획서", "지금", "이유"].map((t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 } } }));
  s.addTable([hdr,
    ["대상 부품", "MVTec 2종 + Real3D-AD 2종", "T-LESS 4종 + MulSen-AD 3종", "MVTec는 위에서 한 장 찍은 스캔뿐, MulSen-AD는 360° + 실제 결함"],
    ["MVTec 3D-AD 역할", "대상 물체 + 센서 차이", "검출기 구현 확인 + 실제 센서 비교", "\"dowel\"이 원기둥 핀이 아니라 벽 앵커"],
    ["설정 하나 평가", "5–15분 (부품 60개)", "약 1분 (부품 218개)", "시점 저장본; 60개로는 1% 기준선 불가"],
    ["특징", "색 + 형상", "형상(깊이)만", "대상이 형상 결함, 깊이는 조명 영향 없음"],
    ["검출기", "메모리 뱅크 1종", "+ 설계도 거리 + 딥러닝 특징", "정직한 비교, 검출기 의존성 확인"],
    ["실험 판정", "기준 없음", "사전 등록", "결과를 본 뒤 기준을 바꾸지 않기 위해"],
    ["공정 가정", "명시 없음", "방향 맞춤 공급(±3°)", "무작위 방향에서 정합이 무너짐"],
    ["주장 범위", "공정 안의 설정 최적화", "같은 목표, \"최초\" 표현 안 씀", "구성 요소마다 선행 연구 존재"],
  ], { x: 0.6, y: 1.35, w: 12.1, colW: [1.8, 2.7, 3.1, 4.5], fontSize: 12.5, color: C.text1, valign: "middle",
       border: { type: "solid", pt: 0.75, color: HEX.grid }, rowH: 0.58, margin: 0.06 });
  s.addNotes("목표와 지표는 그대로입니다. 바뀐 것은 대상 부품과 데이터의 역할, 계산 방법, 검증 장치입니다.");
}
{
  const s = newSlide("CONTENT", "남은 일과 일정");
  const steps = [["10/27", "탐색 효율 · 딥러닝 특징 검출기 비교 완료"], ["10/27", "실제 결함 순위 재실행 · 잡음 없는 센서 비교"],
                 ["10/29", "교차 적용 확대 재실행 → 부품별 최적 대 범용 설정 비교"], ["10/31", "부품 형상 특징–최적 설정 관계 실험"],
                 ["11/3", "중간 점검 · 멘토 피드백"], ["11월", "보강 실험(전역 정합 등), 데모 다듬기"],
                 ["12/15", "기말 발표 · 최종 보고서"], ["2027/3/1", "IEEE CASE 2027 투고 목표"]];
  s.addShape(pres.shapes.LINE, { x: 0.9, y: 3.0, w: 11.6, h: 0, line: { color: C.text2, width: 2 } });
  steps.forEach(([d, t], i) => {
    const x = 0.6 + i * 1.5;
    s.addShape(pres.shapes.OVAL, { x: x + 0.45, y: 2.82, w: 0.36, h: 0.36, fill: { color: i < 4 ? C.accent1 : C.accent2 },
                                   line: { color: C.background1, width: 2 } });
    text(s, d, { x, y: 2.2, w: 1.3, h: 0.45, fontSize: 14, bold: true, align: "center" });
    text(s, t, { x: x - 0.02, y: 3.35, w: 1.36, h: 1.6, fontSize: 12, color: C.text2, align: "center" });
  });
  text(s, "노란 점은 대기열에 들어간 실험. 10/29·10/31 실험은 탐색적 관찰에서 나온 가설을 실행 전에 판정 규칙까지 고정한 새 실험이다. 날짜는 추정.",
       { x: 0.6, y: 5.3, w: 12.1, h: 0.5, fontSize: 13, color: C.text2 });
  s.addNotes("노란 점은 이미 자동으로 실행되도록 예약된 실험이고, 파란 점은 일정입니다. 논문 작성은 지금은 보류하고 결과만 기록하고 있습니다.");
}
{
  const s = newSlide("CONTENT", "멘토님께 여쭙고 싶은 것");
  const Q = ["판정 기준: 실제 라인에서 \"정상품 오판정 1%\"는 적절한가? 0.1%나 \"미검 0\"에 가까운 기준을 쓰는지",
             "비용: 카메라 수·사이클 시간 외에 카메라 단가, 조명, 회전대 장비비, 설치 공간을 넣어야 하는지",
             "센서 기준: 실제 고급 구조광 카메라가 우리 기본보다 훨씬 깨끗했다. 현장 센서 수준(보급형/고급형)은?",
             "부품 공급: 소형 부품은 대부분 방향을 맞춰 공급되는지, 무작위 방향도 다뤄야 하는지",
             "결과 제시: 현장 엔지니어에게는 \"권장 설정 하나\"와 \"성능–비용 표\" 중 어느 쪽이 쓸모 있는지",
             "실물 검증: 실제 카메라나 라인에서 몇 가지 설정을 측정해 볼 산학 협력 기회가 있는지"];
  Q.forEach((q, i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.35 + Math.floor(i / 2) * 1.8;
    card(s, x, y, 5.95, 1.6);
    numberDot(s, i + 1, x + 0.25, y + 0.25, 0.45);
    text(s, q, { x: x + 0.9, y: y + 0.22, w: 4.85, h: 1.25, fontSize: 14.5 });
  });
  s.addNotes("현장 관점의 조언이 필요한 여섯 가지입니다.");
}
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: section });
  s.addText("감사합니다", { placeholder: "title" });
  s.addText([
    { text: "ISCO · Inspection Setup Configuration Optimizer", options: { breakLine: true, color: C.accent1 } },
    { text: "팀 강화실패 · 박정규 · 김상준", options: { breakLine: true } },
    { text: "자료: docs/reports/ (종합 보고서 · 데모 영상 · 그림)", options: {} },
  ], { placeholder: "body" });
  img(s, path.join(STILL, "slots.jpg"), 7.75, 0.75, 5.1, 2.87, "카메라 후보 자리");
  img(s, path.join(STILL, "part_mulsen_nut.jpg"), 7.75, 3.85, 5.1, 2.87, "너트 검사 장면");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (SKILL) {
    const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));
    await applyTheme(OUT, THEME);
  }
  console.log(`[deck] ${OUT}`);
})();
