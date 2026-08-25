# FAQ

<!-- md-trans-meta sourceCommit=4a22b17b608bcc429ba2f8a5211d79cc753fb40f translatedAt=2026-08-12T10:57:23.789Z pushedAt=2026-08-20T11:47:59.796Z -->

## 1.  **What should I do if a red cann-cla/no label appears after I submit a PR?**

This label indicates that among the commits included in the PR, some contributors have not signed the CANN community Contributor License Agreement (CLA). The signing link can be found in the PR comment section.

1) If contributing as an individual, select "Sign Individual CLA".

2) If contributing on behalf of an enterprise, select "Sign Corporate CLA".

3) If contributing as an employee of an enterprise that has already signed the Corporate CLA, select "Corporate Contributor Registration".

- After signing, you will receive an email with the subject "Signing CLA on project of xx". Contact the Corporation Managers listed in the email for approval.

- After approval, comment "/check-cla" in the PR comment section to re-trigger the CLA check, and the cann-cla/yes label will be applied. The CLA check uses the committer email from the commit information as the verification credential. This email can be queried using `git log --pretty=fuller`.

<table>
<tbody><tr>
<th>Scenario</th>
<th>Choice</th>
<th>Solution</th>
</tr>
<tr>
<td>The commit email matches the GitCode submission email</td>
<td>Use this email uniformly</td>
<td>Sign the CLA directly at the signing address above using this email</td>
</tr>
<tr>
<td rowspan="2">The commit email does not match the GitCode submission email</td>
<td>You want to sign using the commit email</td>
<td>Modify the GitCode commit email to match the commit email. Add the commit email on the GitCode personal settings page and set it as the commit email, then complete the CLA signing at the signing address above.</td>
</tr>
<tr>
<td>Want to sign using the GitCode commit email</td>
<td>On the local machine where Git is running, execute the commands <code>git config --global user.name ****</code> and <code>git config --global user.email ****</code> to change the Git commit email to the GitCode commit email. After that, go to the signing address to complete the CLA signing.</td>
</tr>
</tbody>
</table>

## 2.  **Why can't I fork the CANN/abc repository to my personal account?**

This issue usually occurs because a repository with the same name "abc" already exists under your personal account. For example, you may have previously forked a repository named "abc" from the CANN organization. Since GitCode resolves repositories by combining your personal account name with the repository name, having two repositories with the same name under your personal account is not allowed.

Solution: Modify the name and path of the existing repository under your personal account, and then fork the repository from "CANN/abc" again.

## 3.  **What is the difference between protected branches and unprotected branches?**

Protected branches allow you to configure specific roles or members with push and merge permissions for that branch. Unprotected branches do not support this.

## 4. **Can CANN developers directly push code to the repository?**

CANN developers are not allowed to directly push code to the community repository. Only repository administrators can push code to the community repository. If CANN developers want to contribute code, they can only fork the community repository to their personal account and contribute code by submitting a Pull Request.

## 5. **What is the difference between directly pushing code to a repository and merging code via /lgtm and /approve comments?**

Directly pushing code to a repository via git commands lacks the necessary review process and carries certain merge risks. The main app scenario is when files to be uploaded are too large and exceed the personal repository limit, in which case the only option is to directly push to an unprotected branch of the repository and then merge from the unprotected branch to the protected branch.

Merging code via /lgtm and /approve comments adds a review step to the process, ensuring that merging a piece of code requires the review and approval of at least one committer other than the submitter. Even if the submitter is a committer, approval from another committer is still required.

## 6. **What commands are supported in the CANN community repository comment section, and what do they mean?**

For the commands currently supported in the community repository comment section, see [CANN Community Comment Commands](infra-command.md).

## 7.  **What are the reasons and solutions for CI build not being triggered after submitting a PR?**

There are usually two scenarios where continuous integration (CI) is not triggered in time:

- The first possibility is that due to network issues or system task scheduling issues, the webhook notification event sent from the code repository does not reach the target service in time, so the CI build is not triggered. In this case, you can re-trigger the build by entering **/compile** in the PR comment.

- The second possibility is that a PR is submitted shortly after the code repository is created, and the CI build project has not yet been created on the Jenkins server side, so the CI build cannot be triggered, and commenting **/compile** will not take effect either. In this case, please wait for the system to automatically create the build project.
