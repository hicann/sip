# FFT Common Interfaces

<!-- md-trans-meta sourceCommit=cec88e057607a630073cce4bbace3c21f8d93fe7 translatedAt=2026-08-12T10:55:06.492Z pushedAt=2026-08-20T11:47:59.781Z -->

- **asdFftCreate**

    Function description: Registers an FFT handle.\
    Function prototype: AspbStatus asdFftCreate(asdFftHandle &handle)\
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
      <td>handle (asdFftHandle)</td>
      <td>Input</td>
      <td>Handle of the <code>asdFftCreate</code> API.</td>
    </tr>
  </tbody>
    </table>

- **asdFftSetStream**

    Function description: Binds an NPU execution stream.\
    Function prototype: AspbStatus asdFftSetStream(asdFftHandle handle, void *stream)\
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
      <td>handle (asdFftHandle)</td>
      <td>Input</td>
      <td>Handle of the <code>asdFftSetStream</code> API.</td>
    </tr>
    <tr>
      <td>stream (void *)</td>
      <td>Input</td>
      <td>Pointer to the stream object.</td>
    </tr>
  </tbody>
    </table>

- **asdFftDestroy**

    Function description: Destroys the handle and releases the space occupied by the handle.\
    Function prototype: AspbStatus asdFftDestroy(asdFftHandle handle)\
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
      <td>handle (asdFftHandle)</td>
      <td>Input</td>
      <td>Handle of the <code>asdFftDestroy</code> API.</td>
    </tr>
  </tbody>
    </table>

- **asdFftGetWorkspaceSize**

    Function description: Computes the workspace size required by the FFT execution stream under the current plan.\
    Function prototype: AspbStatus asdFftGetWorkspaceSize(asdFftHandle handle, size_t &workSize)\
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
      <td>handle (asdFftHandle)</td>
      <td>Input</td>
      <td>Handle of the <code>asdFftGetWorkspaceSize</code> API.</td>
    </tr>
    <tr>
      <td>workSize (size_t &)</td>
      <td>Input</td>
      <td>Required workspace size.</td>
    </tr>
  </tbody>
    </table>

- **asdFftSetWorkspace**

    Function description: Configures the workspace required for the FFT computation process bound to the current handle.\
    Function prototype: AspbStatus asdFftSetWorkspace(asdFftHandle handle, void *workspace)\
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
      <td>handle (asdFftHandle)</td>
      <td>Input</td>
      <td>Handle of the <code>asdFftSetWorkspace</code> API.</td>
    </tr>
    <tr>
      <td>workspace (void *)</td>
      <td>Input</td>
      <td>Pointer to the workspace.</td>
    </tr>
  </tbody>
    </table>

- **asdFftSynchronize**

    Function description: Synchronizes the NPU status.\
    Function prototype: AspbStatus asdFftSynchronize(asdFftHandle handle)\
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
      <td>handle (asdFftHandle)</td>
      <td>Input</td>
      <td>Handle of the <code>asdFftSynchronize</code> API.</td>
    </tr>
  </tbody>
    </table>

- **asdFftGetType**

    Function description: Returns the type of FFT computation bound to the current handle, including ASCEND_FFT_C2C, ASCEND_FFT_C2R, and ASCEND_FFT_R2C.\
    Function prototype: AspbStatus asdFftGetType(asdFftHandle handle, asdFftType &fftType)\
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
      <td>handle (asdFftHandle)</td>
      <td>Input</td>
      <td>Handle of the <code>asdFftGetType</code> API.</td>
    </tr>
    <tr>
      <td>fftType (asdFftType)</td>
      <td>Input/Output</td>
      <td>Used to receive the FFT type value.</td>
    </tr>
  </tbody>
    </table>
    