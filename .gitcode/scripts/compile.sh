#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

set -ex

if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
    echo "source devtoolset"
    source /opt/rh/devtoolset-7/enable
fi
gcc --version

cd ${WORKSPACE}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/$(arch)-linux/devlib
bash build.sh
ret=$?

package_path=output/Ascend-cann-SIP_linux-$(arch).run
if [ $ret -eq 0 ]; then
    compile_package_name=$(ls ${WORKSPACE}/output/*.run 2>/dev/null | head -n1)
    mv ${compile_package_name} ${WORKSPACE}/${package_path}
    echo package_path=${package_path} >> $ATOMGIT_OUTPUT
fi
echo ret=$ret >> $ATOMGIT_OUTPUT
exit $ret
