//
//  TextValueView.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/24/23.
//

import Cocoa

class TextValueView: NSView {
    
    // MARK: Properties
    
    let rows = 5
    let columns = 4
    var totalCells: Int {
        rows * columns
    }
    
    // MARK: Views
    
    var parentViews: [[NSView]] = []
    var labels:  [[NSTextField]] = []
    var values:  [[NSTextField]] = []
    
    // MARK: Init
    
    init() {
        super.init(frame: .zero)
        setupViews()
        setupValueViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    // MARK: Actions
    
    // MARK: Public
    
    func setLabelValue(index: Int, label: String, value: String) {
        let row = index / (columns)
        let column = index % columns
        labels[row][column].stringValue = label
        values[row][column].stringValue = value
    }
    
    // MARK: Private
    
    private func setupValueViews() {
        var count = 0
        for rowIndex in 0..<rows {
            parentViews.append([])
            labels.append([])
            values.append([])
            for columnIndex in 0..<columns {
                count += 1
                let parentView = NSView()
                parentViews[rowIndex].append(parentView)
                parentView.wantsLayer = true
                parentView.translatesAutoresizingMaskIntoConstraints = false
                addSubview(parentView)
                NSLayoutConstraint.activate([
                    parentView.widthAnchor.constraint(equalTo: widthAnchor, multiplier: 1 / CGFloat(columns)),
                    parentView.heightAnchor.constraint(equalTo: heightAnchor, multiplier: 1 / CGFloat(rows))
                ])
                if rowIndex == 0 {
                    NSLayoutConstraint.activate([
                        parentView.topAnchor.constraint(equalTo: topAnchor)
                    ])
                } else {
                    NSLayoutConstraint.activate([
                        parentView.topAnchor.constraint(equalTo: parentViews[rowIndex - 1][columnIndex].bottomAnchor)
                    ])
                }
                if columnIndex == 0 {
                    NSLayoutConstraint.activate([
                        parentView.leadingAnchor.constraint(equalTo: leadingAnchor)
                    ])
                } else {
                    NSLayoutConstraint.activate([
                        parentView.leadingAnchor.constraint(equalTo: parentViews[rowIndex][columnIndex - 1].trailingAnchor)
                    ])
                }
                let subView = NSView()
                subView.wantsLayer = true
                subView.translatesAutoresizingMaskIntoConstraints = false
                parentView.addSubview(subView)
                NSLayoutConstraint.activate([
                    subView.centerYAnchor.constraint(equalTo: parentView.centerYAnchor),
                    subView.leadingAnchor.constraint(equalTo: parentView.leadingAnchor, constant: 4),
                    subView.trailingAnchor.constraint(equalTo: parentView.trailingAnchor, constant: -4)
                ])
                
                let label = NSTextField(labelWithString: "")
                label.cell?.wraps = true
                label.textColor = CGColor.lightGrayTextColor.NSColor()
                label.font = NSFont(name: "BrassMono-Bold", size: 12)
                subView.addSubview(label)
                label.translatesAutoresizingMaskIntoConstraints = false
                NSLayoutConstraint.activate([
                    label.topAnchor.constraint(equalTo: subView.topAnchor),
                    label.leadingAnchor.constraint(equalTo: subView.leadingAnchor)
                ])
                
                let value = NSTextField(labelWithString: "")
                value.font = NSFont(name: "BrassMono-Bold", size: 16)
                value.textColor = .white
                subView.addSubview(value)
                value.translatesAutoresizingMaskIntoConstraints = false
                NSLayoutConstraint.activate([
                    value.topAnchor.constraint(equalTo: label.bottomAnchor),
                    value.leadingAnchor.constraint(equalTo: subView.leadingAnchor),
                    value.bottomAnchor.constraint(equalTo: subView.bottomAnchor)
                ])
                labels[rowIndex].append(label)
                values[rowIndex].append(value)
            }
        }
    }
    
    // MARK: Setup Views
    
    private func setupViews() {
        
    }
}
