//
//  LabelValueController.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/24/23.
//

import Foundation

class LabelValueController {
    
    private weak var view: TextValueView? = nil
    
    static let shared = LabelValueController()
    
    func setLabelValue(index: Int, label: String, value: String) {
        view?.setLabelValue(index: index, label: label, value: value)
    }
    
    func setView(_ view: TextValueView) {
        self.view = view
    }
}
