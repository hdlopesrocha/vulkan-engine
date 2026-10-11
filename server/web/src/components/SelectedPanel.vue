<template>
  <div class="sec">
    <h2>Selected chunk (click a box)</h2>
    <div v-if="!selected" style="font-size:12px;color:#9ca3af">none — click a box or a row below</div>
    <div v-else class="kv">
      <b>id</b><span>{{ selected.id }}</span>
      <b>layer</b><span>{{ layerName(selected.layer) }} ({{ selected.layer }})</span>
      <b>level</b><span>{{ selected.level }}</span>
      <b>min</b><span>{{ fmt3(selected.minX, selected.minY, selected.minZ) }}</span>
      <b>size</b><span>{{ selected.size }}</span>
      <b>max</b><span>{{ fmt3(selected.minX + selected.size, selected.minY + selected.size, selected.minZ + selected.size) }}</span>
      <b>lod / chunkLod</b><span>{{ selected.lod }} / {{ selected.chunkLod }}</span>
      <b>leaf</b><span>{{ selected.isLeaf ? 'yes' : 'no' }}</span>
      <b>type</b><span>{{ spaceName(selected.spaceType) }}</span>
      <b>version</b><span>{{ selected.version }}</span>
      <b>brush</b><span><span class="swatch" :style="{ background: brushSwatch }"></span>{{ selected.brushIndex }}</span>
      <b>sdf[8]</b><span>{{ selected.sdf.map(fmtF).join(', ') }}</span>
      <b>mesh</b><span>{{ meshStatus }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { selected, layerName, fmt3, fmtF, spaceName, meshStatus, rendererOf } from '../store.js';
import { brushColor } from '../proto.js';

const brushSwatch = computed(() => {
  if (!selected.value) return 'transparent';
  const cc = brushColor(selected.value.brushIndex, rendererOf(selected.value.layer));
  return `rgb(${Math.round(cc[0] * 255)},${Math.round(cc[1] * 255)},${Math.round(cc[2] * 255)})`;
});
</script>
