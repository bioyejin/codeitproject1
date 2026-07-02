# rf-detr/model.py
"""RF-DETR 모델 생성 함수."""

_VARIANTS = {
    'nano': 'RFDETRNano',
    'small': 'RFDETRSmall',
    'medium': 'RFDETRMedium',
    'base': 'RFDETRBase',
    'large': 'RFDETRLarge',
}


def get_rfdetr_model(variant='small', checkpoint_path=None):
    """
    RF-DETR 모델 변형(variant)을 생성합니다.
    rfdetr 패키지는 학습(model.train)/추론(model.predict)을 자체적으로
    수행하므로, src/model.py의 get_model()과 달리 nn.Module forward 계약을
    따르지 않는 별도 인터페이스임에 주의.

    Args:
        variant (str): 'nano' | 'small' | 'medium' | 'base' | 'large'
                        (roboflow/rf-detr 소스의 _VARIANT_EXPORTS에 다섯 클래스 모두 존재 확인함)
        checkpoint_path (str): 주어지면 학습된 가중치를 로드한 상태로 생성합니다.
            RF-DETR 생성자의 pretrain_weights 인자를 사용 (roboflow/rf-detr 소스의
            detr.py에서 model_config.pretrain_weights로 체크포인트를 로드하는 것을 확인함).

    Returns:
        RF-DETR 모델 인스턴스
    """
    import rfdetr

    class_name = _VARIANTS.get(variant.lower())
    if class_name is None:
        raise ValueError(f"알 수 없는 RF-DETR variant: {variant} (지원: {list(_VARIANTS)})")
    if not hasattr(rfdetr, class_name):
        raise ValueError(f"설치된 rfdetr 패키지에 {class_name}이 없음 (variant={variant})")

    model_cls = getattr(rfdetr, class_name)
    if checkpoint_path:
        return model_cls(pretrain_weights=checkpoint_path)
    return model_cls()
