<svg width="100%" viewBox="0 0 680 920" xmlns="http://www.w3.org/2000/svg">
<style>
  @media (prefers-color-scheme: light) {
    :root { --p: #2C2C2A; --s: #5F5E5A; --t: #888780; --bg2: #F1EFE8; --b: rgba(0,0,0,0.15); }
    .c-blue rect, .c-blue circle, .c-blue ellipse { fill: #E6F1FB; stroke: #185FA5; }
    .c-blue .th { fill: #0C447C; } .c-blue .ts { fill: #185FA5; }
    .c-teal rect, .c-teal circle, .c-teal ellipse { fill: #E1F5EE; stroke: #0F6E56; }
    .c-teal .th { fill: #085041; } .c-teal .ts { fill: #0F6E56; }
    .c-coral rect, .c-coral circle, .c-coral ellipse { fill: #FAECE7; stroke: #993C1D; }
    .c-coral .th { fill: #712B13; } .c-coral .ts { fill: #993C1D; }
    .c-purple rect, .c-purple circle, .c-purple ellipse { fill: #EEEDFE; stroke: #534AB7; }
    .c-purple .th { fill: #3C3489; } .c-purple .ts { fill: #534AB7; }
    .c-amber rect, .c-amber circle, .c-amber ellipse { fill: #FAEEDA; stroke: #854F0B; }
    .c-amber .th { fill: #633806; } .c-amber .ts { fill: #854F0B; }
    .c-gray rect, .c-gray circle, .c-gray ellipse { fill: #F1EFE8; stroke: #5F5E5A; }
    .c-gray .th { fill: #444441; } .c-gray .ts { fill: #5F5E5A; }
    .c-pink rect, .c-pink circle, .c-pink ellipse { fill: #FBEAF0; stroke: #993556; }
    .c-pink .th { fill: #72243E; } .c-pink .ts { fill: #993556; }
  }
  @media (prefers-color-scheme: dark) {
    :root { --p: #D3D1C7; --s: #B4B2A9; --t: #888780; --bg2: #2C2C2A; --b: rgba(255,255,255,0.15); }
    .c-blue rect, .c-blue circle, .c-blue ellipse { fill: #0C447C; stroke: #85B7EB; }
    .c-blue .th { fill: #B5D4F4; } .c-blue .ts { fill: #85B7EB; }
    .c-teal rect, .c-teal circle, .c-teal ellipse { fill: #085041; stroke: #5DCAA5; }
    .c-teal .th { fill: #9FE1CB; } .c-teal .ts { fill: #5DCAA5; }
    .c-coral rect, .c-coral circle, .c-coral ellipse { fill: #712B13; stroke: #F0997B; }
    .c-coral .th { fill: #F5C4B3; } .c-coral .ts { fill: #F0997B; }
    .c-purple rect, .c-purple circle, .c-purple ellipse { fill: #3C3489; stroke: #AFA9EC; }
    .c-purple .th { fill: #CECBF6; } .c-purple .ts { fill: #AFA9EC; }
    .c-amber rect, .c-amber circle, .c-amber ellipse { fill: #633806; stroke: #EF9F27; }
    .c-amber .th { fill: #FAC775; } .c-amber .ts { fill: #EF9F27; }
    .c-gray rect, .c-gray circle, .c-gray ellipse { fill: #2C2C2A; stroke: #B4B2A9; }
    .c-gray .th { fill: #D3D1C7; } .c-gray .ts { fill: #B4B2A9; }
    .c-pink rect, .c-pink circle, .c-pink ellipse { fill: #72243E; stroke: #ED93B1; }
    .c-pink .th { fill: #F4C0D1; } .c-pink .ts { fill: #ED93B1; }
  }
  .th { font-family: system-ui, -apple-system, sans-serif; font-size: 14px; font-weight: 500; }
  .ts { font-family: system-ui, -apple-system, sans-serif; font-size: 12px; font-weight: 400; }
  .arr { stroke-width: 1.5; }
  .leader { stroke: var(--t); stroke-width: 0.5; stroke-dasharray: 4 3; }
</style>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
</defs>

<text class="th" x="340" y="28" text-anchor="middle" style="font-size:15px" fill="var(--p)">generate 모듈 — 솔루션 영상 생성 파이프라인</text>
<text class="ts" x="340" y="46" text-anchor="middle" fill="var(--s)">generate.py (Sora-2) / Geminigenerate.py (Veo 3.1)</text>

<!-- Trigger -->
<g class="c-blue">
<rect x="40" y="68" width="180" height="44" rx="8" stroke-width="0.5"/>
<text class="th" x="130" y="90" text-anchor="middle" dominant-baseline="central">Flutter 앱</text>
</g>

<line x1="220" y1="90" x2="268" y2="90" stroke="#1D9E75" stroke-width="0.5" marker-end="url(#arrow)"/>
<text class="ts" x="244" y="82" fill="#1D9E75">POST</text>

<g class="c-teal">
<rect x="270" y="68" width="200" height="44" rx="8" stroke-width="0.5"/>
<text class="th" x="370" y="90" text-anchor="middle" dominant-baseline="central">RAG 서버 /generate-video</text>
</g>

<line x1="470" y1="90" x2="518" y2="90" stroke="#D85A30" stroke-width="0.5" marker-end="url(#arrow)"/>

<g class="c-coral">
<rect x="520" y="68" width="130" height="44" rx="8" stroke-width="0.5"/>
<text class="th" x="585" y="84" text-anchor="middle" dominant-baseline="central">subprocess</text>
<text class="ts" x="585" y="100" text-anchor="middle" dominant-baseline="central">Popen</text>
</g>

<line x1="585" y1="112" x2="585" y2="148" stroke="var(--s)" stroke-width="0.5" marker-end="url(#arrow)"/>

<!-- Step 1 -->
<g class="c-amber">
<rect x="420" y="150" width="220" height="70" rx="8" stroke-width="0.5"/>
<text class="th" x="530" y="174" text-anchor="middle" dominant-baseline="central">1. 대화 컨텍스트 수집</text>
<text class="ts" x="530" y="192" text-anchor="middle" dominant-baseline="central">Firestore collection_group</text>
<text class="ts" x="530" y="208" text-anchor="middle" dominant-baseline="central">→ 최신 메시지 역추적</text>
</g>

<g class="c-amber">
<rect x="100" y="158" width="160" height="56" rx="8" stroke-width="0.5"/>
<text class="th" x="180" y="180" text-anchor="middle" dominant-baseline="central">Firestore</text>
<text class="ts" x="180" y="198" text-anchor="middle" dominant-baseline="central">chat_rooms/{id}/messages</text>
</g>

<line x1="260" y1="185" x2="418" y2="185" stroke="#BA7517" stroke-width="0.5" marker-end="url(#arrow)"/>

<text class="ts" x="420" y="238" fill="var(--s)">실패 시 폴백: 하드코딩된 OE 에러 시나리오</text>

<line x1="530" y1="220" x2="530" y2="260" stroke="var(--s)" stroke-width="0.5" marker-end="url(#arrow)"/>

<!-- Step 2 -->
<g class="c-purple">
<rect x="380" y="262" width="260" height="70" rx="8" stroke-width="0.5"/>
<text class="th" x="510" y="286" text-anchor="middle" dominant-baseline="central">2. 영상 프롬프트 생성</text>
<text class="ts" x="510" y="304" text-anchor="middle" dominant-baseline="central">Gemini 2.5 Flash</text>
<text class="ts" x="510" y="320" text-anchor="middle" dominant-baseline="central">대화 맥락 → 시각적 묘사 프롬프트</text>
</g>

<g class="c-gray">
<rect x="60" y="268" width="280" height="56" rx="8" stroke-width="0.5"/>
<text class="th" x="200" y="288" text-anchor="middle" dominant-baseline="central">프롬프트 지시사항</text>
<text class="ts" x="200" y="306" text-anchor="middle" dominant-baseline="central">세로형 비율 / 3.5~8초 / LG ELLE 워터마크</text>
</g>

<line x1="340" y1="296" x2="378" y2="296" stroke="var(--s)" stroke-width="0.5" stroke-dasharray="4 3" marker-end="url(#arrow)"/>

<line x1="510" y1="332" x2="510" y2="370" stroke="var(--s)" stroke-width="0.5" marker-end="url(#arrow)"/>

<!-- Step 3 -->
<g class="c-purple">
<rect x="60" y="372" width="250" height="70" rx="8" stroke-width="0.5"/>
<text class="th" x="185" y="396" text-anchor="middle" dominant-baseline="central">3-A. OpenAI Sora-2</text>
<text class="ts" x="185" y="414" text-anchor="middle" dominant-baseline="central">videos.create → 폴링 3초 간격</text>
<text class="ts" x="185" y="430" text-anchor="middle" dominant-baseline="central">→ download_content → .mp4</text>
</g>

<g class="c-purple">
<rect x="370" y="372" width="270" height="70" rx="8" stroke-width="0.5"/>
<text class="th" x="505" y="396" text-anchor="middle" dominant-baseline="central">3-B. Google Veo 3.1 (대체)</text>
<text class="ts" x="505" y="414" text-anchor="middle" dominant-baseline="central">generate_videos → 폴링 대기</text>
<text class="ts" x="505" y="430" text-anchor="middle" dominant-baseline="central">→ .save() → .mp4 (9:16, 8초)</text>
</g>

<text class="ts" x="340" y="408" text-anchor="middle" fill="var(--s)">or</text>

<line x1="340" y1="442" x2="340" y2="480" stroke="var(--s)" stroke-width="0.5" marker-end="url(#arrow)"/>

<!-- Step 4 -->
<g class="c-gray">
<rect x="200" y="482" width="280" height="56" rx="8" stroke-width="0.5"/>
<text class="th" x="340" y="504" text-anchor="middle" dominant-baseline="central">4. 파일 저장</text>
<text class="ts" x="340" y="522" text-anchor="middle" dominant-baseline="central">assets_generate/result_solution_{timestamp}.mp4</text>
</g>

<line x1="280" y1="538" x2="160" y2="580" stroke="#1D9E75" stroke-width="0.5" marker-end="url(#arrow)"/>
<line x1="400" y1="538" x2="510" y2="580" stroke="#BA7517" stroke-width="0.5" stroke-dasharray="4 3" marker-end="url(#arrow)"/>

<text class="ts" x="196" y="556" fill="#1D9E75">경로 A (활성)</text>
<text class="ts" x="480" y="556" fill="#BA7517">경로 B (비활성)</text>

<!-- Path A -->
<g class="c-teal">
<rect x="40" y="582" width="240" height="56" rx="8" stroke-width="0.5"/>
<text class="th" x="160" y="604" text-anchor="middle" dominant-baseline="central">5-A. 로컬 URL 생성</text>
<text class="ts" x="160" y="622" text-anchor="middle" dominant-baseline="central">http://{ip}:8000/assets/{file}</text>
</g>

<!-- Path B -->
<g class="c-amber">
<rect x="400" y="582" width="240" height="56" rx="8" stroke-width="0.5"/>
<text class="th" x="520" y="604" text-anchor="middle" dominant-baseline="central">5-B. Firebase Storage 업로드</text>
<text class="ts" x="520" y="622" text-anchor="middle" dominant-baseline="central">chat_rooms/video_00001.mp4</text>
</g>

<line x1="160" y1="638" x2="160" y2="676" stroke="var(--s)" stroke-width="0.5" marker-end="url(#arrow)"/>

<!-- Step 6 -->
<g class="c-amber">
<rect x="60" y="678" width="240" height="56" rx="8" stroke-width="0.5"/>
<text class="th" x="180" y="700" text-anchor="middle" dominant-baseline="central">6. Firestore 메시지 등록</text>
<text class="ts" x="180" y="718" text-anchor="middle" dominant-baseline="central">message_type: "VIDEO" + video_url</text>
</g>

<!-- Watcher -->
<g class="c-coral">
<rect x="380" y="678" width="260" height="70" rx="10" stroke-width="0.5"/>
<text class="th" x="510" y="700" text-anchor="middle" dominant-baseline="central">RAG 서버 watch_new_videos()</text>
<text class="ts" x="510" y="718" text-anchor="middle" dominant-baseline="central">assets_generate/ 3초마다 감시</text>
<text class="ts" x="510" y="736" text-anchor="middle" dominant-baseline="central">새 mp4 감지 → Firestore 자동 등록</text>
</g>

<path d="M480 538 L580 538 L580 676" fill="none" stroke="#D85A30" stroke-width="0.5" stroke-dasharray="4 3" marker-end="url(#arrow)"/>
<text class="ts" x="590" y="610" fill="#D85A30">파일 감지</text>

<line x1="180" y1="734" x2="180" y2="772" stroke="var(--s)" stroke-width="0.5" marker-end="url(#arrow)"/>
<line x1="510" y1="748" x2="510" y2="772" stroke="var(--s)" stroke-width="0.5" stroke-dasharray="4 3" marker-end="url(#arrow)"/>

<!-- Step 7 -->
<g class="c-blue">
<rect x="130" y="774" width="420" height="56" rx="10" stroke-width="0.5"/>
<text class="th" x="340" y="796" text-anchor="middle" dominant-baseline="central">7. Flutter 채팅방에 영상 메시지 표시</text>
<text class="ts" x="340" y="814" text-anchor="middle" dominant-baseline="central">VideoPlayerScreen / VideoPlayerOverlayScreen</text>
</g>

<!-- Polling -->
<g class="c-gray">
<rect x="130" y="850" width="420" height="44" rx="8" stroke-width="0.5"/>
<text class="th" x="340" y="872" text-anchor="middle" dominant-baseline="central">Flutter /check-video-status 폴링 → 완료 시 재생 UI 전환</text>
</g>

<line x1="340" y1="830" x2="340" y2="848" stroke="var(--s)" stroke-width="0.5" marker-end="url(#arrow)"/>

</svg>
