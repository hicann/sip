# Contribution Guide

## Understand the Code of Conduct

SiP is an open project under CANN. Before contributing, please review the [CANN Open Project Code of Conduct](./zh/context/code-of-conduct.md). All subsequent activities in the SiP project (including but not limited to posting comments, submitting issues, and posting wikis) must follow this code of conduct.

## Sign a CLA

Before contributing to the project, you need to sign the CANN Open Project Contributor License Agreement (CLA).

Select the appropriate institutional CLA, institutional contributor CLA, individual CLA, or enterprise administrator CLA based on your participation identity. Click [here](https://clasign.osinfra.cn/sign/68cbd4a3dbabc050b436cdd4) to sign.

- Institutional CLA: Contribute as an enterprise representative. Sign the CLA on behalf of the enterprise. This is typically signed by an enterprise manager.
- Institutional Contributor CLA: If you are an employee of an enterprise that has signed the Institutional CLA, apply to sign the Institutional Contributor CLA. Select your enterprise on the application page. After the application, the enterprise administrator will review it. Once approved, you can participate in contributions.
- Individual CLA: Contribute as an individual who is not an enterprise employee. Sign the Individual CLA.
- Enterprise Administrator: Contribute as an enterprise administrator. Sign the Enterprise Administrator CLA. Enterprise administrators have the authority to review applications for Institutional Contributor CLA and manage personnel.

## Participating in Contributions

After signing the CLA agreement, you can start your contribution journey. There are many ways to contribute, and every contribution is welcome and valued.

You can report, discuss, and track all issues you discover or new ideas you want to contribute through [Issues](#submit-issue--handle-issue-tasks), and close the associated issues after the [contribution coding](#contribution-coding) Pull-Request is merged.

> 📝 **Tip**
>
> - If you encounter issues during the PR submission process, refer to [FAQs](./zh/context/infra-faqs.md) for solutions to common problems.

### Contribution Categories

- Operator Bug Fix

  If you discover certain operator bugs in this repository and want to fix them, you are welcome to create an issue in the repository for tracking and resolution.

  Follow the [Submit Issue / Handle Issue Tasks](#submit-issue--handle-issue-tasks) instructions below to create a `Bug-Report` issue describing the bug,
  then enter "/assign" or "/assign @yourself" in the comment box to assign the issue to yourself for handling.

- Operator Optimization

  If you have ideas for generalization enhancement or performance optimization for certain operator implementations in this repository and want to implement these optimizations, you are welcome to contribute operator optimizations.

  Follow the [Submit Issue / Handle Issue Tasks](#submit-issue--handle-issue-tasks) instructions below to create a `Requirement` issue describing the optimization points and provide your design proposal,
  then enter "/assign" or "/assign @yourself" in the comment box to assign the issue to yourself for tracking the optimization.

- Contributing New Operators

  If you have a brand new operator that you want to design and implement based on Ascend chips, you are welcome to propose new ideas and designs in an issue and discuss them with Ascend team members.

  Follow the [Submit Issue / Handle Issue Tasks](#submit-issue--handle-issue-tasks) instructions below to create a `Requirement` issue providing your new operator description and design proposal,
  Ascend team members will communicate with you for confirmation and provide an appropriate `contrib` directory category for your operator. You can contribute your new operator to the corresponding category directory.

  At the same time, you need to comment "/assign" or "/assign @yourself" in the submitted issue to claim the issue and submit the new operator to the code repository subsequently.

- Documentation Correction

  If you discover errors in operator documentation in the repository, you are welcome to create an issue in the repository for reporting and fixing.

  Follow the [Submit Issue / Handle Issue Tasks](#submit-issue--handle-issue-tasks) instructions below to create a `Documentation` issue pointing out the problems in the corresponding documentation,
  then enter "/assign" or "/assign @yourself" in the comment box to assign the issue to yourself for correcting the corresponding documentation description.

- Helping Others Solve Issues

  If you have appropriate solutions to problems encountered by others in the community, you are welcome to post comments in the issue to help others solve problems and pain points, and jointly improve usability.

  If the corresponding issue requires code modification, you can enter "/assign" or "/assign @yourself" in the issue comment box to assign the issue to yourself and track and assist in solving the problem.

### <a name="Submit Issue"></a>Submit Issue / Handle Issue Tasks

- Find the Issue list:

  On the [SiP](https://gitcode.com/cann/sip) project Gitcode homepage, click "Issues" to find the Issue list.

- Submit an Issue

  If you plan to report a bug to the community, submit a requirement, or contribute your own opinions or suggestions to the community, please submit an [Issue](https://gitcode.com/cann/sip/issues).

- Participate in Issue Discussion

  Each issue supports discussion among developers. If you are interested, you can post your own opinions in the comment section.

- Find an Issue You Are Willing to Handle

  If you are willing to handle one of the issues, you can assign it to yourself. Simply enter "/assign" or "/assign @yourself" in the comment box, and the robot will assign the issue to you. Your name will appear in the responsible person list.

### Contribution Coding

1. Prepare the CANN Development Environment

   If you want to participate in coding contributions, you need to prepare the CANN development environment. For details, refer to [Environment Preparation](../README_en.md#3--environment-setup).

2. Understand SiP Development Precautions

   1) For tool version requirements and installation, refer to [Tool Version Requirements and Installation](../README_en.md#315--tool-version-requirements-and-installation) to understand the environment and tool requirements for coding contributions.

   2) SiP software coding follows the license agreement: CANN Open Software License Agreement Version 2.0. For the detailed agreement description, refer to the [LICENSE](../LICENSE) file. If you contribute code to the SiP source code repository, you must follow this agreement.

     Add the following declaration at the top of newly created source files such as cpp, cc, and h:

     ```txt
     /**
      * Copyright (c) 2025 [Name of the copyright owner] All rights reserved.
      * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
      * CANN Open Software License Agreement Version 2.0 (the "License").
      * Please refer to the License for details. You may not use this file except in compliance with the License.
      * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED,
      * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
      * See LICENSE in the root of the software repository for the full text of the License.
      */
     ```

     Add the following declaration at the top of newly created files such as py and sh:

     ```txt
     # Copyright (c) 2025 [Name of the copyright owner] All rights reserved.
     # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
     # CANN Open Software License Agreement Version 2.0 (the "License").
     # Please refer to the License for details. You may not use this file except in compliance with the License.
     # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED,
     # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
     # See LICENSE in the root of the software repository for the full text of the License.
     ```

    - If you contribute on behalf of yourself and you own the copyright to the contributed content, replace `[Name of the copyright owner]` in the first line with your signature.
    - If you contribute on behalf of your employer, or your employer owns the copyright to the contributed content, replace `[Name of the copyright owner]` in the first line with your employer's name.

      If you have any questions about the copyright ownership of the contributed content, consult a legal advisor or your employer's legal team.

    - `2025` in the first line is the year when you created or modified the file. Modify it according to the actual time.

3. Code Download and Contribution Process
![Code Contribution Process](./zh/API_Reference/figures/docs_images_contrib-flow.png)
   (1) Before starting code development, first fork the required SiP repository to your personal repository, then download the personal repository to your local machine. Make code modifications on a local branch.
   (2) After code validation meets the contribution requirements, submit a Pull-Request to contribute the code to SiP. You can find the submitted Pull-Request in the [Pull-Request list](https://gitcode.com/cann/sip/pulls).
   (3) In the comment section of the submitted Pull-Request, comment `compile` to trigger compilation.
   (4) Pay attention to viewing the gate test results. If the test does not pass, modify the local code based on the problem prompts. If the test passes, the PR will be assigned to a committer for review. Pay attention to the committer's review comments.
   (5) After your PR is reviewed and approved, the code will be merged into the SiP source code repository.
