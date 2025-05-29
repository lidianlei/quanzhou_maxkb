<template>
  <el-dialog v-model="dialogVisible" title="柱形图" width="80%" height="70%" :before-close="handleClose">
    <div ref="chartRef" style="width: 100%; height: 500px"></div>
    <template #footer>
      <div class="dialog-footer">
        <el-button @click="handleClose">关闭</el-button>
      </div>
    </template>
  </el-dialog>
</template>
<script setup lang="ts">
import * as echarts from 'echarts'
import { nextTick, onMounted, ref } from 'vue'

const dialogVisible = ref(false)

let chartRef = ref(null)

const props = defineProps<{
  echartsData: object | any
  chartState: boolean
}>()


const emit = defineEmits(['closeChart'])

onMounted(() => {
  dialogVisible.value = props.chartState;
  nextTick(() => {
    const chart = echarts.init(chartRef.value)
    chart.setOption(props.echartsData)
  })
})

const handleClose = (done: any) => {
  dialogVisible.value = false;
  emit('closeChart')
  done()
}
</script>
<style scoped lang="scss"></style>
