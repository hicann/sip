# BLAS公共接口

## 算子使用说明

若需使用BLAS算子，需先创建句柄，然后调用对应算子的plan接口，初始化该句柄对应的算子配置并进行绑定，接着调用BLAS公共接口“asdBlasGetWorkspaceSize”获取计算所需workspace大小以及包含了算子计算流程的执行器，然后调用“asdBlasSetWorkspace”给对应的plan设置需要的workspace，最后调用BLAS算子接口执行计算。计算完需要对plan进行销毁，以免造成内存泄漏。

## 公共接口说明

- **asdBlasCreate**

    功能描述：创建全局唯一的handle。\
函数原型：AspbStatus asdBlasCreate(asdBlasHandle &handle)\
    **参数说明：**
    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入/输出</td>
      <td>asdBlasCreate接口的句柄。</td>
    </tr>
  </tbody>
    </table>

- **asdBlasSetStream**

    功能描述：将使用runtime创建的stream与具体的plan实例进行绑定。\
    函数原型：AspbStatus asdBlasSetStream(asdBlasHandle handle, void *stream)\
    **参数说明：**
    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>asdBlasSetStream接口的句柄。</td>
    </tr>
    <tr>
      <td>stream（void *）</td>
      <td>输入</td>
      <td>指向流对象的指针。</td>
    </tr>
  </tbody>
    </table>

- **asdBlasDestroy**

    功能描述：销毁创建的plan并释放对应plan申请的资源。\
    函数原型：AspbStatus asdBlasDestroy(asdBlasHandle handle)\
    **参数说明：**
    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>asdBlasDestroy接口的句柄。</td>
    </tr>
  </tbody>
    </table>

- **asdBlasSetWorkspace**

    功能描述：给对应的plan设置所需要的workspace。\
    函数原型：AspbStatus asdBlasSetWorkspace(asdBlasHandle handle, void *workSpace)\
    **参数说明：**
    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>asdBlasSetWorkspace接口的句柄。</td>
    </tr>
    <tr>
      <td>workSpace（void *）</td>
      <td>输入</td>
      <td>指针，指向存储所需的工作空间。</td>
    </tr>
  </tbody>
    </table>

- **asdBlasSynchronize**

    功能描述：同步等待算子执行。\
    函数原型：AspbStatus asdBlasSynchronize(asdBlasHandle handle)\
    **参数说明：**
    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>asdBlasSynchronize接口的句柄。</td>
    </tr>
  </tbody>
    </table>

- **asdBlasGetWorkspaceSize**

    功能描述：计算所需workspace大小以及包含了算子计算流程的执行器。\
    函数原型：AspbStatus asdBlasGetWorkspaceSize(asdBlasHandle handle, size_t &workspaceSize);\
    **参数说明：**
    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle（asdBlasHandle）</td>
      <td>输入</td>
      <td>asdBlasGetWorkspaceSize接口的句柄。</td>
    </tr>
    <tr>
      <td>workspaceSize（size_t &）</td>
      <td>输入/输出</td>
      <td>所需工作空间大小。</td>
    </tr>
  </tbody>
    </table>
