# BLAS Common Interfaces

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:46:33.905Z pushedAt=2026-08-20T11:47:59.709Z -->

## Operator Usage Instructions

To use BLAS operators, first create a handle, then call the plan interface of the corresponding operator to initialize and bind the operator configuration associated with the handle. Next, call the BLAS common interface `asdBlasGetWorkspaceSize` to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then call `asdBlasSetWorkspace` to set the required workspace for the corresponding plan. Finally, call the BLAS operator interface to execute the computation. After the computation is complete, the plan must be destroyed to avoid memory leaks.

## Common Interface Description

- **asdBlasCreate**

  Function description: Creates a globally unique handle.\
  Function prototype: AspbStatus asdBlasCreate(asdBlasHandle &handle)\
  **Parameter description**:

  <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle (asdBlasHandle)</td>
      <td>Input/Output</td>
      <td>Handle of the <code>asdBlasCreate</code> API.</td>
    </tr>
  </tbody>
    </table>

- **asdBlasSetStream**

    Function description: Binds a stream created using the runtime to a specific plan instance.\
    Function prototype: AspbStatus asdBlasSetStream(asdBlasHandle handle, void *stream)\
    Parameter description:

    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle (asdBlasHandle)</td>
      <td>Input</td>
      <td>Handle of the <code>asdBlasSetStream</code> API.</td>
    </tr>
    <tr>
      <td>stream (void *)</td>
      <td>Input</td>
      <td>Pointer to the stream object.</td>
    </tr>
  </tbody>
    </table>

- **asdBlasDestroy**

    Function description: Destroys the created plan and releases the resources allocated for the corresponding plan.\
    Function prototype: AspbStatus asdBlasDestroy(asdBlasHandle handle)\
    **Parameter description:**

    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle (asdBlasHandle)</td>
      <td>Input</td>
      <td>Handle of the <code>asdBlasDestroy</code> API.</td>
    </tr>
  </tbody>
    </table>

- **asdBlasSetWorkspace**

    Function description: Sets the required workspace for the corresponding plan.\
    Function prototype: AspbStatus asdBlasSetWorkspace(asdBlasHandle handle, void *workSpace)\
    **Parameter description:**

    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle (asdBlasHandle)</td>
      <td>Input</td>
      <td>Handle of the <code>asdBlasSetWorkspace</code> API.</td>
    </tr>
    <tr>
      <td>workSpace (void *)</td>
      <td>Input</td>
      <td>Pointer to the storage of the required workspace.</td>
    </tr>
  </tbody>
    </table>

- **asdBlasSynchronize**

    Function description: Synchronously waits for operator execution to complete.\
    Function prototype: AspbStatus asdBlasSynchronize(asdBlasHandle handle)\
    **Parameter description:**

    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle (asdBlasHandle)</td>
      <td>Input</td>
      <td>Handle of the <code>asdBlasSynchronize</code> API.</td>
    </tr>
  </tbody>
    </table>

- **asdBlasGetWorkspaceSize**

    Function description: Computes the required workspace size and the executor that contains the operator computation flow.\
    Function prototype: AspbStatus asdBlasGetWorkspaceSize(asdBlasHandle handle, size_t &workspaceSize);\
    Parameter description:

    <table style="undefined;table-layout: fixed; width: 880px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 510px">
  </colgroup>
  <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>handle (asdBlasHandle)</td>
      <td>Input</td>
      <td>Handle of the <code>asdBlasGetWorkspaceSize</code> API.</td>
    </tr>
    <tr>
      <td>workspaceSize (size_t &)</td>
      <td>Input/Output</td>
      <td>Required workspace size.</td>
    </tr>
  </tbody>
    </table>
