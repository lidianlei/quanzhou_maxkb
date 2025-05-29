const llaMaRouter = {
  path: '/llaMa',
  name: 'llaMa',
  meta: { title:  'views.dataset.llaMa', permission: 'DATASET:READ' },
  component: () => import('@/layout/layout-template/AppLayout.vue'),
  redirect: '/datasetIndex',
  children: [
    {
      path: '/datasetIndex',
      name: 'datasetIndex',
      meta: { title: 'llaMa', activeMenu: '/llaMa' },
      component: () => import('@/views/llaMa/Index.vue')
    },
   
  ]
}

export default llaMaRouter
