<template>
  <div id="app">
    <HeaderBar />
    <div v-if="!connected" class="connwarn">
      <span>WS unreachable: <code>{{ wsTarget }}</code> — {{ wsError }}</span>
      <span>Start the server (<code>cd bin &amp;&amp; ./server</code>, default port 8080) and open
        <code>http://localhost:8080/</code> — not the HTML file directly.
        Overrides: <code>?port=8081</code> or <code>?ws=ws://host:port/ws</code></span>
      <button @click="reconnect">Reconnect</button>
    </div>
    <main>
      <ViewportCanvas />
      <aside>
        <div class="sec">
          <h2>Filters (list + 3D view)</h2>
          <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:12px">
            <label class="chk" v-for="l in layerList" :key="'v' + l"><input type="checkbox" v-model="layerVisibility[l]"> {{ layerName(l) }}</label>
          </div>
          <div style="font-size:11px;color:#9ca3af;margin-top:4px">The per-column search boxes below filter the list and the rendered chunks (numbers accept &gt;, &lt;, ranges like 1-5).</div>
        </div>
        <SelectedPanel />
        <div class="sec list" style="flex:1">
          <ChunkTable />
        </div>
      </aside>
    </main>
    <footer>binary + gzip over ws /ws · magic CHK1 · record 76B · mesh 0x02/0x14/0x16 · {{ wsTarget }} · {{ lastMsg }}</footer>
  </div>
</template>

<script setup>
import { onMounted } from 'vue';
import HeaderBar from './components/HeaderBar.vue';
import ViewportCanvas from './components/ViewportCanvas.vue';
import SelectedPanel from './components/SelectedPanel.vue';
import ChunkTable from './components/ChunkTable.vue';
import {
  connected, wsTarget, wsError, lastMsg, layerList, layerVisibility,
  layerName, connect, reconnect, selectedId, meshStatus, showWireframe,
  renderAll, allMeshes, chunks, colFilter, layerInfo,
} from './store.js';
import { cam } from './camera.js';
import { dbg, installEvictHook } from './gl.js';

installEvictHook();

// Render-state snapshot for automated tests (read-only).
window.__wire = () => ({
  ...dbg,
  meshId: selectedId.value,
  meshStatus: meshStatus.value,
  selectedId: selectedId.value,
  showWireframe: showWireframe.value,
  renderAll: renderAll.value,
  allMeshes: allMeshes.size,
  chunks: chunks.value.size,
  filters: { ...colFilter.value },
  layers: JSON.parse(JSON.stringify(layerInfo.value)),
  cam: { pos: [...cam.pos], quat: [...cam.quat] },
});

onMounted(() => {
  connect();
});
</script>
