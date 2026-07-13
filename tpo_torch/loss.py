import torch
import torch.nn.functional as F


def tpo_loss_from_logits(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 0.1,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    TPO Loss from raw logits using a Pointwise Distribution Shift.

    This implementation solves the 'canceling-out' bug by shifting the log-odds
    of specific tokens, rather than the entire vocabulary vector.
    """
    # 1. Shift for causal LM (predict next token)
    shift_p_logits = policy_logits[..., :-1, :].contiguous()
    shift_r_logits = reference_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    seq_len = shift_p_logits.size(1)
    batch = shift_p_logits.size(0)

    # 2. Extract Log-Probs
    p_lprobs = F.log_softmax(shift_p_logits, dim=-1)
    r_lprobs = F.log_softmax(shift_r_logits, dim=-1)

    # 3. Gather log-probs at label positions
    p_lp_gathered = torch.gather(p_lprobs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    r_lp_gathered = torch.gather(r_lprobs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # 4. Handle Advantage Broadcasting
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1).expand(-1, seq_len)
    elif advantages.size(1) == 1:
        advantages = advantages.expand(-1, seq_len)
    elif advantages.size(1) != seq_len:
        # Pad or truncate advantages to match seq_len
        if advantages.size(1) > seq_len:
            advantages = advantages[:, :seq_len]
        else:
            pad = torch.zeros(batch, seq_len - advantages.size(1), device=advantages.device)
            advantages = torch.cat([advantages, pad], dim=1)

    # 5. Pointwise TPO Math (Numerical Armor)
    # Compute log-odds directly from log-probs without exp/log round-trip:
    #   log(p)  = lse,  log(1-p) = log(1 - exp(lse))
    #   log_odds = log(p/(1-p)) = lse - softplus(-lse) [via log-sigmoid identity]
    #   target = sigmoid(log_odds_ref + advantage/beta)
    with torch.no_grad():
        neg_r = -r_lp_gathered
        log_odds_ref = r_lp_gathered - torch.nn.functional.softplus(neg_r)
        log_odds_ref = log_odds_ref.clamp(min=-30.0, max=30.0)
        target_probs = torch.sigmoid(log_odds_ref + (advantages / beta)).detach()

    # 6. Cross-Entropy Loss
    per_token_loss = -(target_probs * p_lp_gathered)

    # 7. Masking & Averaging
    if attention_mask is not None:
        mask = attention_mask[..., 1:].float()
        active_loss = per_token_loss * mask
        loss = active_loss.sum() / (mask.sum() + 1e-8)
    else:
        loss = per_token_loss.mean()

    return loss


def tpo_loss(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 0.1,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    TPO Loss from pre-gathered log-probabilities.
    """
    if policy_logprobs.dim() == 1:
        policy_logprobs = policy_logprobs.unsqueeze(0)
        reference_logprobs = reference_logprobs.unsqueeze(0)
        advantages = advantages.unsqueeze(0)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0)

    batch, seq_len = policy_logprobs.shape[:2]

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1).expand(-1, seq_len)
    elif advantages.size(1) == 1:
        advantages = advantages.expand(-1, seq_len)

    with torch.no_grad():
        neg_r = -reference_logprobs
        log_odds_ref = reference_logprobs - torch.nn.functional.softplus(neg_r)
        log_odds_ref = log_odds_ref.clamp(min=-30.0, max=30.0)
        target_probs = torch.sigmoid(log_odds_ref + (advantages / beta)).detach()

    per_token_loss = -(target_probs * policy_logprobs)

    if attention_mask is not None:
        mask = attention_mask.float()
        active_loss = per_token_loss * mask
        loss = active_loss.sum() / (mask.sum() + 1e-8)
    else:
        loss = per_token_loss.mean()

    return loss
