<template>
  <h2>All chunks ({{ filtered.length }} shown of {{ chunks.size }})</h2>
  <div class="rows">
    <table>
      <thead><tr>
        <th @click="sortBy('id')">id</th><th @click="sortBy('layer')">layer</th>
        <th @click="sortBy('level')">lvl</th><th @click="sortBy('size')">size</th>
        <th @click="sortBy('version')">ver</th><th @click="sortBy('lod')">lod</th>
        <th @click="sortBy('chunkLod')">chunkLoD</th><th @click="sortBy('brushIndex')">brush</th>
      </tr>
      <tr class="filterrow">
        <th><input type="search" v-model="colFilter.id" placeholder="id…"></th>
        <th><input type="search" v-model="colFilter.layer" placeholder="layer…"></th>
        <th><input type="search" v-model="colFilter.level" placeholder="lvl…"></th>
        <th><input type="search" v-model="colFilter.size" placeholder="size…"></th>
        <th><input type="search" v-model="colFilter.version" placeholder="ver…"></th>
        <th><input type="search" v-model="colFilter.lod" placeholder="lod…"></th>
        <th><input type="search" v-model="colFilter.chunkLod" placeholder="chunkLoD…"></th>
        <th><input type="search" v-model="colFilter.brush" placeholder="brush…"></th>
      </tr></thead>
      <tbody>
        <tr v-for="c in filtered.slice(0, 2000)" :key="c.id" :class="{ sel: selectedId && selectedId === c.id }" @click="select(c.id)">
          <td>{{ shortId(c.id) }}</td><td>{{ layerTag(c.layer) }}</td>
          <td>{{ c.level }}</td><td>{{ c.size.toFixed(0) }}</td>
          <td>{{ c.version }}</td><td>{{ c.lod }}</td><td>{{ c.chunkLod }}</td><td>{{ c.brushIndex }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup>
import { chunks, filtered, colFilter, selectedId, select, sortBy, layerTag, shortId } from '../store.js';
</script>
