# Environment Variables

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:58:53.391Z pushedAt=2026-08-20T11:47:59.829Z -->

After the acceleration library is installed, a process-level environment variable setting script `set_env.sh` is provided to automatically complete environment variable configuration. The setting automatically becomes invalid after the user process exits.

## SiP Environment Variables

- **Basic environment variables:**

  <table style="undefined;table-layout: fixed; width: 650px"><colgroup>
    <col style="width: 250px">
    <col style="width: 400px">
  </colgroup>
  <thead>
      <tr>
        <th>Environment Variable Name</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ASDSIP_HOME_PATH</td>
      <td>File storage path after package installation.</td>
    </tr>
     <tr>
      <td>LD_LIBRARY_PATH</td>
      <td>List of search paths for loading dynamic libraries on Linux systems.</td>
    </tr>
  </tbody>
    </table>

- **SiP-related environment variables:**

  <table style="undefined;table-layout: fixed; width: 650px"><colgroup>
    <col style="width: 250px">
    <col style="width: 400px">
  </colgroup>
  <thead>
      <tr>
        <th>Environment Variable Name</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>ASCEND_PROCESS_LOG_PATH</td>
      <td>Sets the log storage path.</td>
    </tr>
     <tr>
      <td>ASCEND_SLOG_PRINT_TO_STDOUT</td>
      <td>Specifies whether to enable log output. When enabled, logs are directly printed and displayed instead of being saved to a log file.</td>
    </tr>
     <tr>
      <td>ASCEND_GLOBAL_LOG_LEVEL</td>
      <td>Sets the log level for application-level logs and the log level of each module. Only debug logs are supported.</td>
    </tr>
     <tr>
      <td>ASCEND_MODULE_LOG_LEVEL</td>
      <td>Sets the log level of each module for application-level logs. Only debug logs are supported.</td>
    </tr>
  </tbody>
    </table>
    