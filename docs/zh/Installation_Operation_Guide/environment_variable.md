# 环境变量参考

加速库安装完成后，提供进程级环境变量设置脚本“set_env.sh”，以自动完成环境变量设置，用户进程结束后自动失效。

## 信号加速库环境变量说明

- **基础环境变量：**
  <table style="undefined;table-layout: fixed; width: 650px"><colgroup>
    <col style="width: 250px">
    <col style="width: 400px">
  </colgroup>
  <thead>
      <tr>
        <th>环境变量名</th>
        <th>说明</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ASDSIP_HOME_PATH</td>
      <td>软件包安装后文件存储路径。</td>
    </tr>
     <tr>
      <td>LD_LIBRARY_PATH</td>
      <td>Linux系统中加载动态库时的搜索路径列表。</td>
    </tr>
  </tbody>
    </table>

- **信号加速库相关环境变量：**
  <table style="undefined;table-layout: fixed; width: 650px"><colgroup>
    <col style="width: 250px">
    <col style="width: 400px">
  </colgroup>
  <thead>
      <tr>
        <th>环境变量名</th>
        <th>说明</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ASCEND_PROCESS_LOG_PATH</td>
      <td>设置日志保存路径。</td>
    </tr>
     <tr>
      <td>ASCEND_SLOG_PRINT_TO_STDOUT</td>
      <td>是否开启日志打印。开启后，日志将不会保存在log文件中，而是将产生的日志直接打印显示。</td>
    </tr>
     <tr>
      <td>ASCEND_GLOBAL_LOG_LEVEL</td>
      <td>设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。</td>
    </tr>
     <tr>
       <td>ASCEND_MODULE_LOG_LEVEL</td>
       <td>设置应用类日志的各模块日志级别，仅支持调试日志。</td>
     </tr>
 </tbody>
   </table>

- **环境变量长度限制：**

 环境变量值的长度上限为 **12800 字节**。超长的环境变量将被忽略（不影响其它变量），并在日志中输出 WARN 级别提示。若发现设置的环境变量未生效，请检查是否超出长度限制。
