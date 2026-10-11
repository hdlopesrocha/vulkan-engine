<template>
  <header>
    <h1>Octree chunks</h1>
    <span class="dot" :class="{ on: connected }"></span>
    <span class="pill">{{ connected ? 'ws connected' : 'ws disconnected' }}</span>
    <span class="pill">total {{ chunks.size }}</span>
    <span class="pill" v-for="l in layerList" :key="'c' + l">{{ layerName(l) }} {{ layerCounts[l] || 0 }}</span>
    <span class="pill">{{ fps.toFixed(0) }} fps</span>
    <span class="pill">{{ culled }} culled</span>
    <button @click="requestAll">Request all</button>
    <button class="ghost" @click="clearAll">Clear</button>
    <button class="ghost" @click="resetCamera" title="Restore the default start view">Reset camera</button>
    <label class="chk"><input type="checkbox" v-model="showWireframe"> wireframe</label>
    <label class="chk" title="Draws every chunk geometry pushed by the server — no requests, purely local"><input type="checkbox" v-model="renderAll" @change="toggleRenderAll"> render all</label>
  </header>
</template>

<script setup>
import {
  chunks, connected, layerList, layerCounts, layerName, fps, culled,
  requestAll, clearAll, showWireframe, renderAll, toggleRenderAll,
} from '../store.js';
import { resetCamera } from '../camera.js';
</script>
